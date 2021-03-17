mkdir -p data

# download preprocessed ConceptNet 5.6.0
mkdir -p data/cpnet/
wget -nc -P data/cpnet/ https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
cd data/cpnet/
yes n | gzip -d conceptnet-assertions-5.6.0.csv.gz
cd ../../
wget -nc -P data/cpnet/ https://csr.s3-us-west-1.amazonaws.com/conceptnet.en.csv
wget -nc -P data/cpnet/ https://csr.s3-us-west-1.amazonaws.com/concept.txt
wget -nc -P data/cpnet/ https://csr.s3-us-west-1.amazonaws.com/relation.txt
wget -nc -P data/cpnet/ https://csr.s3-us-west-1.amazonaws.com/conceptnet.en.pruned.graph

# download pretrained relation embeddings (TransE edge features)
wget -nc -P data/cpnet/ https://csr.s3-us-west-1.amazonaws.com/glove.transe.sgd.rel.npy

# download pretrained entity embeddings (BERT-based	node features provided by Zhengwei)
wget -nc -P data/cpnet/ https://csr.s3-us-west-1.amazonaws.com/tzw.ent.npy

# download CommensenseQA dataset
mkdir -p data/csqa/
wget -nc -P data/csqa/ https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl
wget -nc -P data/csqa/ https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl
wget -nc -P data/csqa/ https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl
wget -nc -P data/csqa/ https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl
wget -nc -P data/csqa/ https://raw.githubusercontent.com/INK-USC/MHGRN/master/data/csqa/inhouse_split_qids.txt

# create output folders
mkdir -p data/csqa/statement/
mkdir -p data/csqa/grounded/
mkdir -p data/csqa/graph/
mkdir -p data/csqa/hybrid/
