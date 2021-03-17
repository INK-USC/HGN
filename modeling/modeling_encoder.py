from transformers import AutoModel, AutoConfig
from transformers import (OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP)
from utils.data_utils import get_gpt_token_num
from utils.layers import *

MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'albert': list(ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'lstm': ['lstm'],
}

MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}


class LSTMTextEncoder(nn.Module):
    pool_layer_classes = {'mean': MeanPoolLayer, 'max': MaxPoolLayer}

    def __init__(self, vocab_size=1, emb_size=300, hidden_size=300, output_size=300, num_layers=2, bidirectional=True,
                 emb_p=0.0, input_p=0.0, hidden_p=0.0, pretrained_emb_or_path=None, freeze_emb=True,
                 pool_function='max', output_hidden_states=False):
        super().__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.output_hidden_states = output_hidden_states
        assert not bidirectional or hidden_size % 2 == 0

        if pretrained_emb_or_path is not None:
            if isinstance(pretrained_emb_or_path, str):  # load pretrained embedding from a .npy file
                pretrained_emb_or_path = torch.tensor(np.load(pretrained_emb_or_path), dtype=torch.float)
            emb = nn.Embedding.from_pretrained(pretrained_emb_or_path, freeze=freeze_emb)
            emb_size = emb.weight.size(1)
        else:
            emb = nn.Embedding(vocab_size, emb_size)
        self.emb = EmbeddingDropout(emb, emb_p)
        self.rnns = nn.ModuleList([nn.LSTM(emb_size if l == 0 else hidden_size,
                                           (hidden_size if l != num_layers else output_size) // (2 if bidirectional else 1),
                                           1, bidirectional=bidirectional, batch_first=True) for l in range(num_layers)])
        self.pooler = self.pool_layer_classes[pool_function]()

        self.input_dropout = nn.Dropout(input_p)
        self.hidden_dropout = nn.ModuleList([RNNDropout(hidden_p) for _ in range(num_layers)])

    def forward(self, inputs, lengths):
        """
        inputs: tensor of shape (batch_size, seq_len)
        lengths: tensor of shape (batch_size)
        returns: tensor of shape (batch_size, hidden_size)
        """
        assert (lengths > 0).all()
        batch_size, seq_len = inputs.size()
        hidden_states = self.input_dropout(self.emb(inputs))
        all_hidden_states = [hidden_states]
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dropout)):
            hidden_states = pack_padded_sequence(hidden_states, lengths, batch_first=True, enforce_sorted=False)
            hidden_states, _ = rnn(hidden_states)
            hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True, total_length=seq_len)
            all_hidden_states.append(hidden_states)
            if l != self.num_layers - 1:
                hidden_states = hid_dp(hidden_states)
        pooled = self.pooler(all_hidden_states[-1], lengths)
        assert len(all_hidden_states) == self.num_layers + 1
        outputs = (all_hidden_states[-1], pooled)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        return outputs


class AttentionMerge(nn.Module):
    """
    https://github.com/jessionlin/csqa/blob/e761b4c5306edf8031a016a391c8f6d93ec3401f/model/layers.py
    H (B, L, hidden_size) => h (B, hidden_size)
    """
    def __init__(self, input_size, attention_size, dropout_prob):
        super(AttentionMerge, self).__init__()
        self.attention_size = attention_size
        self.hidden_layer = nn.Linear(input_size, self.attention_size)
        self.query_ = nn.Parameter(torch.Tensor(self.attention_size, 1))
        self.dropout = nn.Dropout(dropout_prob)

        self.query_.data.normal_(mean=0.0, std=0.02)

    def forward(self, values, mask=None):
        """
        (b, l, h) -> (b, h)
        """
        if mask is None:
            mask = torch.zeros_like(values)
            # mask = mask.data.normal_(mean=0.0, std=0.02)
        else:
            mask = (1 - mask.unsqueeze(-1).type(torch.float)) * -1000.

        keys = self.hidden_layer(values)
        keys = torch.tanh(keys)
        query_var = torch.var(self.query_)
        # (b, l, h) + (h, 1) -> (b, l, 1)
        attention_probs = keys @ self.query_ / math.sqrt(self.attention_size * query_var)
        # attention_probs = keys @ self.query_ / math.sqrt(self.attention_size)

        attention_probs = F.softmax(attention_probs * mask, dim=1)
        attention_probs = self.dropout(attention_probs)

        context = torch.sum(attention_probs + values, dim=1)
        return context


class TextEncoder(nn.Module):
    valid_model_types = set(MODEL_CLASS_TO_NAME.keys())

    def __init__(self, model_name, encoder_pooler='module_pooler', output_token_states=False, from_checkpoint=None, use_segment_id=True, aristo_path=None):
        super().__init__()
        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.encoder_pooler = encoder_pooler
        self.output_token_states = output_token_states
        self.use_segment_id = use_segment_id
        assert not self.output_token_states or self.model_type in ('bert', 'roberta',)

        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        if encoder_pooler == 'att':
            print('use att pooler')
            self.att_merge = AttentionMerge(config.hidden_size, 1024, 0.1)
        self.module = AutoModel.from_pretrained(model_name, config=config)
        if aristo_path is not None:
            print('Loading weights for AristoRoberta...')
            weight = torch.load(aristo_path, map_location='cpu')
            new_dict = {}
            for k, v in weight.items():
                nk = k.replace('_transformer_model.', '')
                if nk not in self.module.state_dict():
                    print(k)
                    continue
                new_dict[nk] = v
            model_dict = self.module.state_dict()
            model_dict.update(new_dict)
            self.module.load_state_dict(model_dict)

        if from_checkpoint is not None:
            self.module = self.module.from_pretrained(from_checkpoint, output_hidden_states=True)
        if self.model_type in ('gpt',):
            self.module.resize_token_embeddings(get_gpt_token_num())
        self.sent_dim = self.module.config.n_embd if self.model_type in ('gpt',) else self.module.config.hidden_size

    def forward(self, *inputs, layer_id=-1):
        '''
        layer_id: only works for non-LSTM encoders
        output_token_states: if True, return hidden states of specific layer and attention masks
        use_segment_id: if False, do not use token_type_ids and do not output token states
        '''

        if self.model_type in ('lstm',):  # lstm
            input_ids, lengths = inputs
            outputs = self.module(input_ids, lengths)
        elif self.model_type in ('gpt',):  # gpt
            input_ids, cls_token_ids, lm_labels = inputs  # lm_labels is not used
            outputs = self.module(input_ids)
        else:  # bert / xlnet / roberta
            if self.use_segment_id:
                input_ids, attention_mask, token_type_ids, output_mask = inputs
                outputs = self.module(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            else:
                assert not self.output_token_states
                input_ids, attention_mask = inputs
                outputs = self.module(input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs[-1]
        hidden_states = all_hidden_states[layer_id]

        if self.output_token_states:
            return hidden_states, output_mask
        # pooling
        if self.model_type in ('lstm',):
            sent_vecs = outputs[1]
        elif self.model_type in ('gpt',):
            cls_token_ids = cls_token_ids.view(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
            sent_vecs = hidden_states.gather(1, cls_token_ids).squeeze(1)
        elif self.model_type in ('xlnet',):
            sent_vecs = hidden_states[:, -1]
        else:  # bert / roberta /albert
            if self.encoder_pooler == 'module_pooler':
                if self.model_type in ('albert',):
                    sent_vecs = self.module.pooler_activation(self.module.pooler(hidden_states[:, 0]))
                else:
                    sent_vecs = self.module.pooler(hidden_states)
            elif self.encoder_pooler == 'cls':
                sent_vecs = hidden_states[:, 0]
            elif self.encoder_pooler == 'mean':
                sent_vecs = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1).unsqueeze(-1)
            else:  # 'att'
                sent_vecs = self.att_merge(hidden_states, attention_mask)
        return sent_vecs, all_hidden_states


def run_test():
    encoder = TextEncoder('lstm', vocab_size=100, emb_size=100, hidden_size=200, num_layers=4)
    input_ids = torch.randint(0, 100, (30, 70))
    lenghts = torch.randint(1, 70, (30,))
    outputs = encoder(input_ids, lenghts)
    assert outputs[0].size() == (30, 200)
    assert len(outputs[1]) == 4 + 1
    assert all([x.size() == (30, 70, 100 if l == 0 else 200) for l, x in enumerate(outputs[1])])
    print('all tests are passed')