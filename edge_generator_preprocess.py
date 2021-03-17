import csv
from collections import Counter
import random

import configargparse
from tqdm import tqdm

parser = configargparse.ArgumentParser()
parser.add_argument('--config', is_config_file=True, help='Config file path.')
parser.add_argument("--input_cpnet_path", default='./data/cpnet/conceptnet-assertions-5.6.0.csv')
parser.add_argument("--output_all_csv_path", default='./data/cpnet/head_tail_sent_all.csv')
parser.add_argument("--output_train_csv_path", default='./data/cpnet/head_tail_sent_train.csv')
parser.add_argument("--output_dev_csv_path", default='./data/cpnet/head_tail_sent_dev.csv')
args = parser.parse_args()

random.seed(42)

# defined in ConceptNet 5.5 paper
sym_rel = {'antonym', 'distinctfrom', 'etymologicallyrelatedto', 'locatednear', 'relatedto', 'similarto', 'synonym'}

need_reverse = ['hassubevent', 'hasfirstsubevent', 'haslastsubevent', 'occupation', 'language', 'capital']
mapping = {
    # 'relatedto',  # is related to (too general)
    # 'formof',  # (grammar-level)
    # 'derivedfrom',  # is derived from (grammar-level)
    'hascontext': 'has context',
    'isa': 'is type of',
    # 'synonym',  # (too general)
    'usedfor': 'is used for',
    # 'etymologicallyrelatedto',  # (grammar-level)
    'similarto': 'is similar to',
    'atlocation': 'is located at',
    'hassubevent': 'is subevent of',
    'hasprerequisite': 'requires',
    'capableof': 'is capable of',
    # 'antonym',  # (too general)
    'causes': 'causes',
    'partof': 'is part of',
    'mannerof': 'is way of',
    'motivatedbygoal': 'is motivated by',
    'hasproperty': 'has property',
    'receivesaction': 'can be',
    'hasa': 'has',
    'causesdesire': 'makes people want',
    'genre': 'is in genre of',
    'hasfirstsubevent': 'is subevent of',
    'distinctfrom': 'is distinct from',
    'desires': 'wants',
    'genus': 'is species of',
    # 'notdesires',  # (not)
    'haslastsubevent': 'is subevent of',
    'definedas': 'is defined as',
    'instanceof': 'is instance of',
    'influencedby': 'is influenced by',
    'occupation': 'is occupation of',
    'language': 'is language of',
    'field': 'is in field of',
    'knownfor': 'is known for',
    'madeof': 'is made of',
    'product': 'makes product',
    'capital': 'is capital of',
    'entails': 'entails',
    # 'notcapableof',  # (not)
    # 'nothasproperty',  # (not)
    'createdby': 'is created by',
    'leader': 'is led by',
    # 'etymologicallyderivedfrom',  # (grammar-level)
    'locatednear': 'is near',
    'symbolof': 'symbolizes'
}

rel_counter = Counter()
all_triples = set()

with open(args.input_cpnet_path, 'r', encoding='utf-8') as csv_file:
    nrow = sum(1 for _ in csv_file)
    csv_file.seek(0)
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for assert_format, rel, head, tail, json_format in tqdm(csv_reader, total=nrow):
        if head.startswith('/c/en/') and tail.startswith('/c/en/'):
            rel = rel.split("/")[-1].lower()
            head = head.split("/")[3].replace('_', ' ').lower()  # Remove part-of-speech (may appear at -1) info
            tail = tail.split("/")[3].replace('_', ' ').lower()
            all_triples.add((head, rel, tail))
            if rel in sym_rel:
                all_triples.add((tail, rel, head))
            rel_counter.update([rel])

print(f'{len(all_triples)} facts found.')
print(rel_counter.most_common())

all_head_tail_sent = set()

for head, rel, tail in all_triples:
    if rel not in mapping:
        continue
    else:
        if rel in need_reverse:
            head, tail = tail, head
        rel = mapping[rel]
        sent = f'{head} {rel} {tail}'
        all_head_tail_sent.add((head, tail, sent))


def write_csv(path, lst):
    with open(path, 'w', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for sample in lst:
            csv_writer.writerow(sample)


all_head_tail_sent = list(all_head_tail_sent)
random.shuffle(all_head_tail_sent)
ten_percent = int(len(all_head_tail_sent) * 0.1)
train_head_tail_sent = all_head_tail_sent[ten_percent:]
dev_head_tail_sent = all_head_tail_sent[:ten_percent]

write_csv(args.output_all_csv_path, all_head_tail_sent)
write_csv(args.output_train_csv_path, train_head_tail_sent)
write_csv(args.output_dev_csv_path, dev_head_tail_sent)
