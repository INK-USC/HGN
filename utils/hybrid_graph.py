import json
import pickle

import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.sparse import coo_matrix

cpnet = None
cpnet_simple = None
concept2id = None
id2concept = None


def get_cpnet_simple(cpnet):
    cpnet_simple = nx.DiGraph()  # without reversed relation
    for u, v, data in cpnet.edges(data=True):
        rel_id = data['rel']
        if rel_id >= 17:
            continue
        if cpnet_simple.has_edge(u, v):  # only keep one relation for an edge
            continue
        else:
            cpnet_simple.add_edge(u, v, weight=rel_id)
    return cpnet_simple


def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = get_cpnet_simple(cpnet)


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}


def get_non_adj_cpts_qa_pair(qcs, acs):
    non_adj_cpts = set()
    adj_cpts = set()
    for qc in qcs:
        for ac in acs:
            if qc != ac:
                qc_id = concept2id[qc]
                ac_id = concept2id[ac]
                has_q2a = cpnet_simple.has_edge(qc_id, ac_id)
                has_a2q = cpnet_simple.has_edge(ac_id, qc_id)
                if has_q2a and has_a2q:
                    adj_cpts.add((qc, ac, cpnet_simple[qc_id][ac_id]['weight']))
                    adj_cpts.add((ac, qc, cpnet_simple[ac_id][qc_id]['weight']))
                elif has_q2a:
                    wt = cpnet_simple[qc_id][ac_id]
                    adj_cpts.add((qc, ac, wt['weight']))
                    adj_cpts.add((ac, qc, wt['weight'] + 17))
                elif has_a2q:
                    wt = cpnet_simple[ac_id][qc_id]
                    adj_cpts.add((ac, qc, wt['weight']))
                    adj_cpts.add((qc, ac, wt['weight'] + 17))
                else:
                    non_adj_cpts.add((qc, ac))
                    non_adj_cpts.add((ac, qc))
    return list(non_adj_cpts), list(adj_cpts)


def find_non_adj_cpts(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path):
    global cpnet, cpnet_simple
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    global concept2id, id2concept
    if any(x is None for x in [concept2id, id2concept]):
        load_resources(cpnet_vocab_path)

    with open(grounded_path, 'r') as fin:
        data = [json.loads(line) for line in fin]
    with open(output_path, 'w') as output_handle:
        num_non_adj = []
        num_adj = []
        for line in tqdm(data):
            non_adj_cp_pairs, adj_cp_pairs = get_non_adj_cpts_qa_pair(line['qc'], line['ac'])
            output_dict = {'question': line['sent'], 'choice': line['ans'], "non_adj_cp_pair": non_adj_cp_pairs, "adj_cp_pair": adj_cp_pairs}
            num_non_adj.append(len(non_adj_cp_pairs))
            num_adj.append(len(adj_cp_pairs))
            output_handle.write(json.dumps(output_dict))
            output_handle.write("\n")
        print(f'#non_adj: {np.mean(num_non_adj)}')
        print(f'#adj: {np.mean(num_adj)}')
        print(output_path)


def generate_pk(extracted_graph_adj_pk, cp_pairs_json_path, cpnet_vocab_path, output_hybrid_graph_adj_pk):
    global concept2id, id2concept
    if any(x is None for x in [concept2id, id2concept]):
        load_resources(cpnet_vocab_path)

    with open(extracted_graph_adj_pk, 'rb') as handle:
        all_qa_pair_data = pickle.load(handle)
    cids = []
    ctypes = []
    for mhgrn_adj, all_cid_lst, q_mask, a_mask in all_qa_pair_data:
        qa_cid_lst = []
        qa_ctype_lst = []
        for cid, is_q, is_a in zip(all_cid_lst, q_mask, a_mask):
            if is_q or is_a:
                qa_cid_lst.append(cid)
                qa_ctype_lst.append(int(is_a))
        cids.append(qa_cid_lst)
        ctypes.append(qa_ctype_lst)

    with open(cp_pairs_json_path, 'r') as fin:
        cp_pairs_data = [json.loads(line) for line in fin]
    assert len(cp_pairs_data) == len(cids) == len(ctypes)
    gen_rel_emb_pointer = 34
    adj_data = []
    for cid_lst, ctype_lst, pair_data_dict in zip(tqdm(cids), ctypes, cp_pairs_data):  # cid_lst: concept_id_lst
        cpnet_id_to_local_id = {w: k for k, w in enumerate(cid_lst)}
        n_node = len(cid_lst)
        if n_node == 0:
            print('Empty graph found: 0 node.')

        extracted_triples = pair_data_dict['adj_cp_pair']
        n_extracted_triples = len(extracted_triples)

        generated_triples = pair_data_dict['non_adj_cp_pair']
        n_generated_triples = len(generated_triples)

        row = np.zeros((n_extracted_triples + n_generated_triples,), dtype=int)
        col = np.zeros((n_extracted_triples + n_generated_triples,), dtype=int)
        rel_emd_id = np.zeros((n_extracted_triples + n_generated_triples,), dtype=int)

        edge_id = 0
        for subj, obj, rel_id in extracted_triples:
            row[edge_id] = cpnet_id_to_local_id[concept2id[subj]]
            col[edge_id] = cpnet_id_to_local_id[concept2id[obj]]
            rel_emd_id[edge_id] = rel_id
            edge_id = edge_id + 1
        for subj, obj in generated_triples:
            row[edge_id] = cpnet_id_to_local_id[concept2id[subj]]
            col[edge_id] = cpnet_id_to_local_id[concept2id[obj]]
            rel_emd_id[edge_id] = gen_rel_emb_pointer  # id of relation
            edge_id = edge_id + 1  # move to next edge
            gen_rel_emb_pointer = gen_rel_emb_pointer + 1
        adj = coo_matrix((rel_emd_id, (row, col)), shape=(n_node, n_node))
        adj_data.append((adj, cid_lst, ctype_lst))
    with open(output_hybrid_graph_adj_pk, 'wb') as fout:
        pickle.dump(adj_data, fout)

