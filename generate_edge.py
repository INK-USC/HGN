#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

import json
import configargparse
import logging
import numpy as np
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import trange

from utils.generator import search
from utils.utils import check_path

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer)
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


parser = configargparse.ArgumentParser()
parser.add_argument('--config', is_config_file=True, help='Config file path.')
parser.add_argument("--generator_ckpt_folder", type=str, default='./saved_edge_generator/best')
parser.add_argument("--input_non_adj_pairs_jsonl", type=str, required=True)
parser.add_argument("--keys", default=['non_adj_cp_pair'], nargs='+', help='The key in input jsonl that stores concept pairs of interest')

parser.add_argument("--output_gen_rel_jsonl", type=str, required=True)
parser.add_argument("--output_pt", type=str, required=True)

parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--search_max_len", type=int, default=10)
parser.add_argument("--encode_max_len", type=int, default=24)
parser.add_argument("--encode_layer", type=int, default=-1)

parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
args = parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

check_path(args.output_gen_rel_jsonl)

set_seed(args)

# Initialize the model and tokenizer
model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer

tokenizer = tokenizer_class.from_pretrained(args.generator_ckpt_folder)
model = model_class.from_pretrained(args.generator_ckpt_folder, output_hidden_states=True)
model.to(args.device)
model.eval()
feature_size = model.config.hidden_size
args.search_max_len = adjust_length_to_model(args.search_max_len, max_sequence_length=model.config.max_position_embeddings)
logger.info(args)


# generate jsonl

def normalize(pair_lst):
    ret_pair_lst = []
    for pair in pair_lst:
        ret_pair = [s.replace('_', ' ') for s in pair[:2]]
        ret_pair_lst.append(ret_pair)
    return ret_pair_lst


all_dics = []
with open(args.input_non_adj_pairs_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        dic = json.loads(line)
        for key in args.keys:
            dic[key] = normalize(dic[key])
        all_dics.append(dic)

flattened_pairs = []
for dic in all_dics:
    for key in args.keys:
        flattened_pairs += dic[key]
flattened_prompts = [subj + tokenizer.sep_token + obj + tokenizer.cls_token for subj, obj in flattened_pairs]
with torch.no_grad():
    flattened_generations = search(flattened_prompts, model, tokenizer, args.search_max_len, args.batch_size, args.device, num_beams=args.num_beams)

output_dics = []
start_idx = 0

for dic in all_dics:
    concept_pairs = []
    for key in args.keys:
        concept_pairs += dic[key]
    end_idx = start_idx + len(concept_pairs)
    generations = flattened_generations[start_idx: end_idx]
    dic['generation'] = generations
    output_dics.append(dic)
    start_idx = end_idx
assert start_idx == len(flattened_generations)
with open(args.output_gen_rel_jsonl, 'w', encoding='utf-8') as relational_f:
    for dic in output_dics:
        relational_f.write(json.dumps(dic) + '\n')


# encode generation as pt


def read_evidence(keys, input_jsonl_path):
    all_evidence = []
    all_evidence_num = []
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            evidence_lst = []
            dic = json.loads(line)
            cp_pairs = []
            for key in keys:
                cp_pairs += dic[key]
            for (subj, obj), sentence in zip(cp_pairs, dic['generation']):
                evidence = subj + tokenizer.sep_token + obj + tokenizer.cls_token + sentence + tokenizer.eos_token
                evidence_lst.append(evidence)
            all_evidence += evidence_lst
            all_evidence_num.append(len(evidence_lst))
    return all_evidence, all_evidence_num


print(f'Reading from {args.output_gen_rel_jsonl}...')
all_evidence, all_evidence_num = read_evidence(args.keys, args.output_gen_rel_jsonl)
print(sum(all_evidence_num))
encoded = tokenizer.batch_encode_plus(all_evidence, add_special_tokens=False, max_length=args.encode_max_len, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
all_input_ids, all_attention_mask = encoded['input_ids'], encoded['attention_mask']
feature_tensor = torch.zeros(len(all_evidence), feature_size)
for start_idx in trange(0, len(all_evidence), args.batch_size, desc='Calculating features...'):
    end_idx = start_idx + args.batch_size
    input_ids = all_input_ids[start_idx: end_idx].to(args.device)
    attention_mask = all_attention_mask[start_idx: end_idx].to(args.device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    all_hidden_states = outputs[-1]
    hidden_states = all_hidden_states[args.encode_layer]
    evidence_len = attention_mask.sum(-1)
    evidence_vecs = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / evidence_len.unsqueeze(1)
    feature_tensor[start_idx: end_idx] = evidence_vecs
output_dic = {'all_evidence_vecs': feature_tensor, 'all_evidence_num': all_evidence_num}
print(f'Saving to {args.output_pt}...')
torch.save(output_dic, args.output_pt)
