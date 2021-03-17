import torch
from collections import defaultdict
from tqdm import tqdm, trange
from multiprocessing import Pool
from multiprocessing import cpu_count


def search(prompt_sequences, model, tokenizer, gen_max_length, batch_size, device, num_beams, num_return_sequences=1):
    """
    prompt_sequences: a list of strings as prompts
    num_beams: 1: greedy search; > 1: beam search
    :return: a list of generated strings
    """
    model.eval()
    if num_beams == 1:
        generated_sequences = []
        for i in trange(0, len(prompt_sequences), batch_size, desc='Greedy Searching...'):
            batched_prompts = prompt_sequences[i: i + batch_size]
            returned_dic = tokenizer.batch_encode_plus(batched_prompts, add_special_tokens=False,
                                                       pad_to_max_length=True, return_tensors="pt")
            input_ids, attention_mask = returned_dic['input_ids'], returned_dic['attention_mask']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output_sequences = greedy_search(input_ids, attention_mask, model, tokenizer, gen_max_length)
            for generated_sequence in output_sequences:
                text = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                generated_sequences.append(text)
    else:
        with Pool(cpu_count()) as p:
            tokenized_inputs = list(
                tqdm(p.imap(tokenizer.encode, prompt_sequences, chunksize=100), total=len(prompt_sequences), desc='Tokenizing...'))
        indices_and_results_by_length = defaultdict(lambda: [[], []])
        for idx, input in enumerate(tokenized_inputs):
            indices_and_results_by_length[len(input)][0].append(idx)
        print('Input lengths distribution:')
        for length, indices_and_results in sorted(indices_and_results_by_length.items(), key=lambda x: x[0]):
            idx_lst = indices_and_results[0]
            print(f'{length:3}: {len(idx_lst):10}')
        print(f'{"Sum":3}: {len(tokenized_inputs):10}')
        print()
        generated_sequences = [None for _ in range(len(tokenized_inputs))]
        for length, indices_and_results in sorted(indices_and_results_by_length.items(), key=lambda x: x[0], reverse=True):
            idx_lst = indices_and_results[0]
            input_sequences = [tokenized_inputs[idx] for idx in idx_lst]
            for i in trange(0, len(input_sequences), batch_size, desc=f'(Len={length}) Beam Searching...'):
                batched_prompts = torch.LongTensor(input_sequences[i: i + batch_size]).to(device)
                output_sequences = beam_search(batched_prompts, model, gen_max_length, num_beams, num_return_sequences)
                for generated_sequence in output_sequences:
                    text = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    indices_and_results[1].append(text)
        for length, indices_and_results in indices_and_results_by_length.items():
            idx_lst, result_lst = indices_and_results
            assert len(idx_lst) == len(result_lst)
            for idx, result in zip(idx_lst, result_lst):
                generated_sequences[idx] = result
        assert all([s is not None for s in generated_sequences])
    return generated_sequences


def beam_search(input_ids, model, gen_max_length, num_beams, num_return_sequences):
    """
    There may be a bug at line 1356 of [PYTHON_ENV_FOLDER]/lib/python3.7/site-packages/transformers/modeling_utils.py
    Have been fixed.
    """
    seq_len = input_ids.size(1)
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=seq_len + gen_max_length,
        do_sample=False,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
    )
    return output_ids[:, seq_len:].tolist()


def greedy_search(input_ids, attention_mask, model, tokenizer, gen_max_length):
    device = input_ids.device
    # credit to https://github.com/huggingface/transformers/issues/3021#issuecomment-591418233
    seq_len = input_ids.size(1)

    num_tokens_to_produce = gen_max_length + seq_len
    eos_not_in_sents = torch.ones(input_ids.size(0)).long().to(device)

    # we need to get the token ids of the last non-padded value
    last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
    start_idx = (last_non_masked_idx).view(-1, 1).repeat(1, len(tokenizer.get_vocab())).unsqueeze(1)

    # get correct position ids
    position_ids = torch.tensor([list(range(seq_len)) for i in range(input_ids.size(0))]).to(device)
    for i, position_ids_slice in enumerate(position_ids):
        position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]]

    for step in range(num_tokens_to_produce):
        outputs = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)

        # in the first decoding step, we want to use the 'real' last position for each sentence
        if step == 0:
            next_token_logits = outputs[0].gather(1, start_idx).squeeze(1)
        else:
            next_token_logits = outputs[0][:, -1, :]

        next_tokens = torch.argmax(next_token_logits, dim=-1)

        # either append a padding token here if <EOS> has been seen or append next token
        tokens_to_add = next_tokens * (eos_not_in_sents) + tokenizer.pad_token_id * (1 - eos_not_in_sents)

        # Update input_ids, attention_mask and position_ids
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1)).long().to(device)],
                                   dim=1)
        position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

        # this updates which sentences have not seen an <EOS> token so far
        # if one <EOS> token was seen the sentence is finished
        eos_not_in_sents.mul_(next_tokens.ne(tokenizer.eos_token_id).long())
        if eos_not_in_sents.max().item() == 0:
            break
    return input_ids[:, seq_len:].tolist()
