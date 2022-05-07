# Learning Contextualized Knowledge Structures for Commonsense Reasoning

Code for paper "Learning Contextualized Knowledge Structures for Commonsense Reasoning" (Findings of ACL'21):

```bibtex
@inproceedings{yan2021learning,
 address = {Online},
 author = {Yan, Jun and Raman, Mrigank and Chan, Aaron and Zhang, Tianyu and Rossi, Ryan  and Zhao, Handong and Kim, Sungchul and Lipka, Nedim and Ren, Xiang},
 booktitle = {Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
 doi = {10.18653/v1/2021.findings-acl.354},
 pages = {4038--4051},
 publisher = {Association for Computational Linguistics},
 title = {Learning Contextualized Knowledge Structures for Commonsense Reasoning},
 url = {https://aclanthology.org/2021.findings-acl.354},
 year = {2021}
}
```

The code is based on [MHGRN](https://github.com/INK-USC/MHGRN/). We thank the authors for open-sourcing their code.

## Requirement

- A new conda environment

> conda create -n HGN python=3.7
>
> conda activate HGN

- PyTorch 1.6.0 + CUDA 10.1

> conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch

- PyTorch Geometric

> pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
>
> pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
>
> pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
>
> pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
>
> pip install torch-geometric

- Transformers 3.1.0, spaCy 2.3.5, ConfigArgParse, NLTK

> pip install transformers==3.1.0 spacy==2.3.5 ConfigArgParse nltk 
> 
> python -m spacy download en


## Data Preparation

```bash
git clone https://github.com/INK-USC/HGN.git
cd HGN
bash download.sh
```

The script will:
- download preprocessed ConceptNet with pretrained entity and relation embeddings;
- download CommonsenseQA dataset;
- create folders for storing the preprocessed dataset.

## Edge Generator

1. Build the training/dev data based on ConceptNet facts.

    ```bash
    python edge_generator_preprocess.py
    ```

2. Train a generator.

    ```bash
    python edge_generator_train.py --config config/edge_generator_train.yaml
    ```
   
## HGN

1. Preprocess the dataset.

    ```bash
    python preprocess.py
    ```
   
    The script will generate for each QA pair:
    - grounded question and answer concepts;
    - an extracted KG subgraph;
    - a jsonl that stores the adj and non-adj concept pairs;
    - a hybrid graph structure (`.pk`) without generated features.
  
2. Generate features (`.pt`) for non-adj concept pairs with the generator.

    It takes the non-adj concept pairs for each instance as input and generate features (`.pt`) with a jsonl as an intermediate output.

    ```bash
    python generate_edge.py --batch_size 100 \
        --input_non_adj_pairs_jsonl ./data/csqa/hybrid/train_cpt_pairs_1hop_hybrid.jsonl \
        --output_gen_rel_jsonl ./data/csqa/hybrid/relgen/train_cpt_pairs_1hop_hybrid.gen.jsonl \
        --output_pt ./data/csqa/hybrid/relgen/train_cpt_pairs_1hop_hybrid.jsonl.pt
    python generate_edge.py --batch_size 100 \
        --input_non_adj_pairs_jsonl ./data/csqa/hybrid/dev_cpt_pairs_1hop_hybrid.jsonl \
        --output_gen_rel_jsonl ./data/csqa/hybrid/relgen/dev_cpt_pairs_1hop_hybrid.gen.jsonl \
        --output_pt ./data/csqa/hybrid/relgen/dev_cpt_pairs_1hop_hybrid.jsonl.pt
    python generate_edge.py --batch_size 100 \
        --input_non_adj_pairs_jsonl ./data/csqa/hybrid/test_cpt_pairs_1hop_hybrid.jsonl \
        --output_gen_rel_jsonl ./data/csqa/hybrid/relgen/test_cpt_pairs_1hop_hybrid.gen.jsonl \
        --output_pt ./data/csqa/hybrid/relgen/test_cpt_pairs_1hop_hybrid.jsonl.pt
    ```

3. Train a HGN model with RoBERTa-large as the text encoder.

    ```bash
    python hgn.py --config config/csqa_roberta.yaml
    ```
    
    Hyperparameters are specified in `config/csqa_roberta.yaml`. The final checkpoint will be saved under `save_dir`. Dev and test accuracy will be printed.

4. (Optional) Output predictions on the test set using a trained HGN model.

    ```bash
    python hgn.py --config config/csqa_roberta.yaml --mode pred \
        --test_model_path <PATH_TO_CHECKPOINT_PT> \
        --output_pred_path <PATH_TO_PREDICTION>
    ```
