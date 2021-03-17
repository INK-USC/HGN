# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import glob
import logging
import os
import pickle
import random
import re
import math
import csv
from typing import Dict, List, Tuple
from datetime import datetime

import configargparse
import numpy as np
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from utils.utils import batch_slice_mask


logger = logging.getLogger(__name__)
parser = configargparse.ArgumentParser()

# Required parameters
parser.add_argument('--config', is_config_file=True, help='Config file path.')

parser.add_argument(
    "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    required=True,
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
    "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
)

# Other parameters
parser.add_argument(
    "--eval_data_file",
    default=None,
    type=str,
    help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
)
parser.add_argument(
    "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in ckpt_dir"
)
parser.add_argument(
    "--model_name",
    default=None,
    type=str,
    help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
)

parser.add_argument(
    "--config_name",
    default=None,
    type=str,
    help="Optional pretrained config name or path if not the same as model_name. If both are None, initialize a new config.",
)
parser.add_argument(
    "--tokenizer_name",
    default=None,
    type=str,
    help="Optional pretrained tokenizer name or path if not the same as model_name. If both are None, initialize a new tokenizer.",
)
parser.add_argument(
    "--max_seq_len",
    default=-1,
    type=int,
    help="Optional input sequence length after tokenization."
         "The training dataset will be truncated in block of this size for training."
         "Default to the model max input length for single sentence inputs (take into account special tokens).",
)
parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
    "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
)
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
)
parser.add_argument(
    "--overwrite_ckpt_dir", action="store_true", help="Overwrite the content of the output directory"
)
parser.add_argument(
    "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
)
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--patience", type=int, default=2, help="early stopping")
parser.add_argument("--train_with_eval_mask", action="store_true")

parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
args = parser.parse_args()

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, f'{args.model_name}_{args.max_seq_len}_{filename}.cache')
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", file_path)
            with open(file_path, 'r', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                lines = []
                for subj, obj, sent in csv_reader:
                    line = subj + tokenizer.sep_token + obj + tokenizer.cls_token + sent + tokenizer.eos_token  # no space should be added
                    lines.append(line)
            lines = [line for line in lines if len(tokenizer.tokenize(line)) <= args.max_seq_len]
            examples = tokenizer.batch_encode_plus(lines, max_length=args.max_seq_len, pad_to_max_length=True, return_tensors='pt')
            examples['attention_mask'] = examples['attention_mask'].bool()
            cls_indices = torch.nonzero(examples['input_ids'] == tokenizer.cls_token_id, as_tuple=True)[1] + 1
            eos_indices = torch.nonzero(examples['input_ids'] == tokenizer.eos_token_id, as_tuple=True)[1] + 1
            eval_mask = batch_slice_mask(examples['input_ids'], cls_indices, eos_indices)
            examples['eval_mask'] = eval_mask
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.input_ids = examples['input_ids']
        self.attention_mask = examples['attention_mask']
        self.eval_mask = examples['eval_mask']

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, i):
        return {
            'input_ids': self.input_ids[i],
            'attention_mask': self.attention_mask[i],
            'eval_mask': self.eval_mask[i]
        }


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.ckpt_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def train(args, train_dataset, model, tokenizer) -> Tuple[int, float]:
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader)) + 1
    else:
        t_total = len(train_dataloader) * args.num_train_epochs

    model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name
        and os.path.isfile(os.path.join(args.model_name, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name and os.path.exists(args.model_name):
        # set global_step to gobal_step of last saved checkpoint from model path
        checkpoint_suffix = args.model_nam.split("-")[-1].split("/")[0]
        global_step = int(checkpoint_suffix)
        epochs_trained = global_step // (len(train_dataloader))
        steps_trained_in_current_epoch = global_step % (len(train_dataloader))

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    best_dev_ppl = 1e6
    best_global_step = 0

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            inputs = batch['input_ids']
            if args.train_with_eval_mask:
                label_mask = batch['eval_mask']
            else:
                label_mask = batch['attention_mask']
            labels = inputs.clone()
            labels[~label_mask] = -100
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                logging.info(f'setp: {global_step} | lr: {scheduler.get_lr()[0]} | train ppl: {math.exp((tr_loss - logging_loss) / args.logging_steps)}')
                logging_loss = tr_loss
                dev_ppl = evaluate(args, model, tokenizer)
                logging.info(f'dev ppl: {dev_ppl}')
                # Save model checkpoint
                ckpt_dir = os.path.join(args.ckpt_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

                torch.save(args, os.path.join(ckpt_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", ckpt_dir)

                torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", ckpt_dir)
                if dev_ppl < best_dev_ppl:
                    best_dev_ppl = dev_ppl
                    ckpt_dir = os.path.join(args.ckpt_dir, "best")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    torch.save(args, os.path.join(ckpt_dir, "training_args.bin"))
                    logger.info("Saving best model checkpoint to %s", ckpt_dir)
                    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
                    logger.info("Saving best optimizer and scheduler states to %s", ckpt_dir)
                    best_global_step = global_step
                if (global_step - best_global_step) / args.logging_steps == args.patience:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if (global_step - best_global_step) / args.logging_steps == args.patience:
            train_iterator.close()
            break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step


def evaluate(args, model, tokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_ckpt_dir = args.ckpt_dir

    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)

    os.makedirs(eval_ckpt_dir, exist_ok=True)

    # Note that DistributedSampler samples randomly

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, eval_mask = batch['input_ids'], batch['eval_mask']
        labels = inputs.clone()
        labels[~eval_mask] = -100
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = math.exp(eval_loss)
    return perplexity


def main(args):
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --ckpt_dir.")
        else:
            args.model_name = sorted_checkpoints[-1]

    if (
        os.path.exists(args.ckpt_dir)
        and os.listdir(args.ckpt_dir)
        and args.do_train
        and not args.overwrite_ckpt_dir
        and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_ckpt_dir to overcome.".format(
                args.ckpt_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Setup logging
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.ckpt_dir, f'{datetime.now().strftime("%m%d_%H%M%S.%f")}.log')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger.warning(
        "device: %s, 16-bits training: %s",
        device,
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name:
        config = AutoConfig.from_pretrained(args.model_name)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    elif args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if args.max_seq_len <= 0:
        args.max_seq_len = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.max_seq_len = min(args.max_seq_len, tokenizer.max_len)

    if args.model_name:
        model = AutoModelWithLMHead.from_pretrained(args.model_name, config=config)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)
    special_tokens_dict = {'sep_token': '<SEP>', 'cls_token': '<CLS>'}

    """
    Input format:
        bank<SEP>door<CLS>bank has door<EOS>
        people<SEP>drive<CLS>person like driving<EOS>
    """

    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = '!'  # with id 0

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoints = [args.ckpt_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.ckpt_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelWithLMHead.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main(args)