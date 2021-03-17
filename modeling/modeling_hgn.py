from torch_scatter import scatter_mean, scatter_sum
from torch_scatter.composite import scatter_softmax
from torch_geometric.data import Data, DataLoader as PyGDataLoader

from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *


def open_graph(graph_path, tensor_path, ablation):
    d = torch.load(tensor_path)
    extraction_emb = torch.zeros(34, 768)
    all_emb = torch.cat([extraction_emb, d['all_evidence_vecs']], dim=0)
    with open(graph_path, 'rb') as handle:
        graph_list = pickle.load(handle)
    data_list = []
    max_rel_id = 0
    for adj_coo_matrix, cid_lst, ctype_lst in tqdm(graph_list):  # each q-a pair has a graph
        row = adj_coo_matrix.row
        col = adj_coo_matrix.col
        rel_id = adj_coo_matrix.data
        if len(rel_id) > 0:  # some instance has no answer concept -> no q-a edge
            max_rel_id = max(max(rel_id), max_rel_id)
        if 'extraction_only' in ablation:
            extraction_mask = rel_id < 34
            row = row[extraction_mask]
            col = col[extraction_mask]
            rel_id = rel_id[extraction_mask]

        rel_id = torch.LongTensor(rel_id)
        edge_index = torch.LongTensor(np.concatenate((row.reshape(1, -1), col.reshape(1, -1)), axis=0))  # 2 * num_of_edges
        edge_attr = torch.cat([rel_id.view(-1, 1), all_emb[rel_id]], dim=1)  # concat rel id with rel emb for each edge; num_of_edges * (1 + gen_rel_emd_dim)

        x = torch.LongTensor(cid_lst)
        node_type = torch.LongTensor(ctype_lst)
        num_of_nodes = torch.LongTensor([len(cid_lst)])
        num_of_edges = torch.LongTensor([len(edge_attr)])  # len(edge_attr) != adj_coo_matrix.nnz for some graphs when 'extraction_only' in ablation
        d = Data(x=x, node_type=node_type, edge_index=edge_index, edge_attr=edge_attr, num_of_nodes=num_of_nodes, num_of_edges=num_of_edges)
        data_list.append(d)
    assert max_rel_id == all_emb.size(0) - 1
    return data_list


class GraphDataLoader(object):
    def __init__(self, train_adj_path, train_gen_pt_path, dev_adj_path, dev_gen_pt_path, test_adj_path, test_gen_pt_path, batch_size, eval_batch_size, num_choice, ablation):
        self.num_choice = num_choice
        self.qa_pair_batch_size = num_choice * batch_size
        self.eval_qa_pair_batch_size = num_choice * eval_batch_size
        self.train_graph_lst = open_graph(train_adj_path, train_gen_pt_path, ablation)
        self.dev_graph_lst = open_graph(dev_adj_path, dev_gen_pt_path, ablation)
        self.has_official_test = test_adj_path is not None
        if self.has_official_test:
            self.test_graph_lst = open_graph(test_adj_path, test_gen_pt_path, ablation)

    def get_pyg_loader(self, indexes_in_train, stats_only=False):
        shuffled_graph_lst = []
        for i in indexes_in_train:
            shuffled_graph_lst += self.train_graph_lst[self.num_choice * i: self.num_choice * (i + 1)]
        avg_node_num = np.mean([d.num_nodes for d in shuffled_graph_lst])
        avg_edge_num = np.mean([d.num_edges for d in shuffled_graph_lst])
        if stats_only:
            return avg_node_num, avg_edge_num  # should be eval bsz for test
        else:
            return PyGDataLoader(shuffled_graph_lst, batch_size=self.qa_pair_batch_size), avg_node_num, avg_edge_num  # should be eval bsz for test

    def dev_graph_data(self):
        avg_node_num = np.mean([d.num_nodes for d in self.dev_graph_lst])
        avg_edge_num = np.mean([d.num_edges for d in self.dev_graph_lst])
        return PyGDataLoader(self.dev_graph_lst, batch_size=self.eval_qa_pair_batch_size), avg_node_num, avg_edge_num

    def test_graph_data(self):
        avg_node_num = np.mean([d.num_nodes for d in self.test_graph_lst])
        avg_edge_num = np.mean([d.num_edges for d in self.test_graph_lst])
        return PyGDataLoader(self.test_graph_lst, batch_size=self.eval_qa_pair_batch_size), avg_node_num, avg_edge_num


class TestGraphDataLoader(object):
    def __init__(self, test_adj_path, test_gen_pt_path, eval_batch_size, num_choice, ablation):
        self.num_choice = num_choice
        self.eval_qa_pair_batch_size = num_choice * eval_batch_size
        self.test_graph_lst = open_graph(test_adj_path, test_gen_pt_path, ablation)

    def test_graph_data(self):
        avg_node_num = np.mean([d.num_nodes for d in self.test_graph_lst])
        avg_edge_num = np.mean([d.num_edges for d in self.test_graph_lst])
        return PyGDataLoader(self.test_graph_lst, batch_size=self.eval_qa_pair_batch_size), avg_node_num, avg_edge_num


class EdgeModel(torch.nn.Module):
    def __init__(self, edge_in_dim, hidden_dim, edge_out_dim, edge_weight_dropout):
        super(EdgeModel, self).__init__()
        self.edge_mlp = MLP(edge_in_dim, hidden_dim, edge_out_dim,
                            2, 0.2, batch_norm=False, layer_norm=True)
        self.weight_mlp = MLP(256, hidden_dim, 1,  # todo: avoid hard-coded numbers
                              1, edge_weight_dropout, batch_norm=False, layer_norm=True)
        self.wt_transform = nn.Linear(edge_in_dim - 128, 128)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, dest, edge_attr, u, edge_batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[edge_batch]], 1)

        out_1 = torch.cat([edge_attr, u[edge_batch]], 1)
        wts = self.weight_mlp(out_1)  # wts: [#edges, 1]
        unnormalized_wts = wts
        wts = scatter_softmax(wts.squeeze(1), edge_batch, dim=0)
        normalized_wts = wts.unsqueeze(1)
        return self.edge_mlp(out), unnormalized_wts, normalized_wts


class NodeModel(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim_1, hidden_dim_2, sent_vec_dim):
        super(NodeModel, self).__init__()
        mlp_1_in_dim = node_in_dim + edge_in_dim
        mlp_2_in_dim = 256 + node_in_dim + sent_vec_dim
        self.message_mlp = MLP(mlp_1_in_dim, hidden_dim_1, 256,
                               2, 0.2, batch_norm=False, layer_norm=True)
        self.node_mlp = MLP(mlp_2_in_dim, hidden_dim_2, 128,
                            2, 0.2, batch_norm=False, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u, node_batch, edge_batch, wts):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        edge_message = torch.cat([x[row], edge_attr], dim=1)
        edge_message = self.message_mlp(edge_message)  # edge_message: [#edges, hidden_dim]
        if wts is None:
            received_message = scatter_mean(edge_message, col, dim=0, dim_size=x.size(0))
        else:
            received_message = scatter_mean(edge_message * wts, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, received_message, u[node_batch]], dim=1)
        return self.node_mlp(out)


class GraphNetwork(torch.nn.Module):
    def __init__(self, edge_model, node_model, ablation):
        super(GraphNetwork, self).__init__()
        self.ablation = ablation
        self.edge_model = edge_model
        self.node_model = node_model

    def forward(self, x, edge_index, edge_attr, u, node_batch):
        """"""
        row, col = edge_index
        edge_batch = node_batch[row]
        edge_attr, unnormalized_wts, normalized_wts = self.edge_model(x[row], x[col], edge_attr, u, edge_batch)
        unnormalized_wts = torch.sigmoid(unnormalized_wts)
        if 'no_edge_weight' in self.ablation:
            x = self.node_model(x, edge_index, edge_attr, u, node_batch, edge_batch, None)
        else:
            if 'unnormalized_edge_weight' in self.ablation:
                x = self.node_model(x, edge_index, edge_attr, u, node_batch, edge_batch, unnormalized_wts)
            else:
                x = self.node_model(x, edge_index, edge_attr, u, node_batch, edge_batch, normalized_wts)

        return x, edge_attr, u, unnormalized_wts, normalized_wts


class GAT(torch.nn.Module):
    def __init__(self, edge_model, node_model):
        super(GAT, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def forward(self, x, edge_index, edge_attr, u, node_batch):
        """"""
        row, col = edge_index
        edge_batch = node_batch[row]
        edge_attr, unnormalized_wts, global_normalized_wts = self.edge_model(x[row], x[col], edge_attr, u, edge_batch)
        # unnormalized_wts = torch.sigmoid(unnormalized_wts)
        local_normalized_wts = scatter_softmax(unnormalized_wts, col, dim=0)
        x = self.node_model(x, edge_index, edge_attr, u, node_batch, edge_batch, local_normalized_wts)

        return x, edge_attr, u, unnormalized_wts, global_normalized_wts


class Decoder(nn.Module):
    def __init__(self, concept_num, concept_dim, relation_num, relation_dim, sent_dim, concept_in_dim,
                 hidden_size, num_attention_heads, fc_size, num_fc_layers, dropout, edge_weight_dropout,
                 pretrained_concept_emb=None, pretrained_relation_emb=None,
                 freeze_ent_emb=True, num_layers=1, ablation=None, emb_scale=1.0):

        super().__init__()

        self.rel_emb = nn.Embedding(relation_num, relation_dim)
        if pretrained_relation_emb is not None:
            self.rel_emb.weight.data.copy_(pretrained_relation_emb)

        self.concept_emb = CustomizedEmbedding(concept_num=concept_num, concept_out_dim=concept_in_dim,
                                               concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb,
                                               freeze_ent_emb=freeze_ent_emb,
                                               scale=emb_scale)
        self.ctype_transform = TypedLinear(concept_in_dim, concept_dim, 2)
        self.ext_transform = nn.Linear(relation_dim + 2 * concept_dim, hidden_size)
        self.gen_transform = nn.Linear(768, hidden_size)

        self.u_transform = nn.Linear(sent_dim, hidden_size)

        edge_model = EdgeModel(edge_in_dim=concept_dim * 2 + hidden_size * 2, hidden_dim=hidden_size, edge_out_dim=hidden_size, edge_weight_dropout=edge_weight_dropout)
        node_model = NodeModel(node_in_dim=concept_dim, edge_in_dim=hidden_size, hidden_dim_1=hidden_size, hidden_dim_2=hidden_size, sent_vec_dim=hidden_size)

        self.ablation = ablation
        if 'GAT' in ablation:
            self.graph_network = GAT(edge_model, node_model)
        else:
            self.graph_network = GraphNetwork(edge_model, node_model, ablation)

        self.attention = MultiheadAttPoolLayer(num_attention_heads, sent_dim, hidden_size)
        self.activation = GELU()
        self.dropout_m = nn.Dropout(dropout)
        self.hid2out = MLP(2*hidden_size + sent_dim, fc_size, 1, num_fc_layers, dropout, batch_norm=False, layer_norm=True)

    def forward(self, sent_vecs, edge_index, c_ids, c_types, node_batch, rel_ids_embs, num_of_nodes, num_of_edges, max_node_num=50):
        row, col = edge_index
        edge_batch = node_batch[row]

        qa_pair_batch_size = num_of_nodes.shape[0]

        node_attr = self.concept_emb(c_ids)
        node_attr = self.ctype_transform(node_attr, c_types)

        rel_ids = rel_ids_embs[:, 0].long()  # [n_edge]
        extraction_rel_mask = rel_ids < 34
        rel_ids = rel_ids * extraction_rel_mask
        rel_ext_embs = self.ext_transform(torch.cat((node_attr[row], self.rel_emb(rel_ids), node_attr[col]), dim=1))
        rel_ext_embs = extraction_rel_mask.unsqueeze(1) * rel_ext_embs

        rel_gen_embs = rel_ids_embs[:, 1:]
        rel_gen_embs = self.gen_transform(rel_gen_embs)
        rel_gen_embs = ~extraction_rel_mask.unsqueeze(1) * rel_gen_embs
        edge_attr = rel_ext_embs + rel_gen_embs

        u = sent_vecs.clone().detach()
        u = self.u_transform(u)
        if 'wo_statement_vec' in self.ablation:
            u = u * 0
        node_vecs, edge_vecs, global_vecs, unnormalized_wts, normalized_wts = self.graph_network(node_attr, edge_index, edge_attr, u, node_batch)

        pooled_edge_vecs = scatter_sum(edge_vecs * normalized_wts, edge_batch, dim=0, dim_size=qa_pair_batch_size)

        evidence_vecs = torch.zeros(qa_pair_batch_size, max_node_num, 128, device=c_ids.device)
        j = 0
        for i in range(qa_pair_batch_size):
            visible_num_tuples = min(num_of_nodes[i].item(), max_node_num)
            evidence_vecs[i, : visible_num_tuples, :] = node_vecs[j: j + visible_num_tuples, :]
            j = j + num_of_nodes[i].item()
        evidence_vecs = self.activation(evidence_vecs)
        mask = torch.arange(max_node_num, device=c_ids.device) >= num_of_nodes.unsqueeze(1)
        mask[mask.all(1), 0] = 0
        pooled_node_vecs, att_scores = self.attention(sent_vecs, evidence_vecs, mask)
        logits = self.hid2out(self.dropout_m(torch.cat((pooled_edge_vecs, pooled_node_vecs, sent_vecs), 1)))
        return logits, unnormalized_wts, normalized_wts


class LMGraphNet(nn.Module):
    def __init__(self, model_name, encoder_pooler,
                 concept_num, concept_dim, relation_num, relation_dim, concept_in_dim, hidden_size,
                 num_attention_heads, fc_size, num_fc_layers, dropout, edge_weight_dropout, pretrained_concept_emb=None,
                 pretrained_relation_emb=None, freeze_ent_emb=True, num_layers=1, ablation=None, emb_scale=1.0,
                 aristo_path=None):

        super().__init__()
        self.encoder = TextEncoder(model_name, encoder_pooler, aristo_path=aristo_path)
        self.decoder = Decoder(concept_num, concept_dim, relation_num, relation_dim, self.encoder.sent_dim, concept_in_dim,
                               hidden_size, num_attention_heads, fc_size, num_fc_layers, dropout, edge_weight_dropout,
                               pretrained_concept_emb, pretrained_relation_emb,
                               freeze_ent_emb=freeze_ent_emb, num_layers=num_layers, ablation=ablation, emb_scale=emb_scale)

    def forward(self, *lm_inputs, edge_index, c_ids, c_types, node_batch, rel_ids_embs, num_of_nodes, num_of_edges):
        batch_size, num_choice = lm_inputs[0].size(0), lm_inputs[0].size(1)
        lm_inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in lm_inputs]  # merge the batch dimension and the num_choice dimension
        sent_vecs, all_hidden_states = self.encoder(*lm_inputs)
        logits, unnormalized_wts, normalized_wts = self.decoder(sent_vecs, edge_index, c_ids, c_types, node_batch, rel_ids_embs, num_of_nodes, num_of_edges)
        logits = logits.view(batch_size, num_choice)
        return logits, unnormalized_wts, normalized_wts


class LMDataLoader(object):
    def __init__(self, train_jsonl_path, dev_jsonl_path, test_jsonl_path,
                 batch_size, eval_batch_size, device, model_name, max_seq_length=128,
                 is_inhouse=True, inhouse_train_qids_path=None, subset_qids_path=None, format=[]):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.is_inhouse = is_inhouse
        self.has_official_test = test_jsonl_path is not None
        self.train_on_subset = subset_qids_path is not None

        model_type = MODEL_NAME_TO_CLASS[model_name]
        self.train_qids, self.train_labels, *self.train_data = load_input_tensors(train_jsonl_path, model_type, model_name, max_seq_length, format=format)
        self.dev_qids, self.dev_labels, *self.dev_data = load_input_tensors(dev_jsonl_path, model_type, model_name, max_seq_length, format=format)
        assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_data)
        assert all(len(self.dev_qids) == x.size(0) for x in [self.dev_labels] + self.dev_data)

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])
        elif self.has_official_test:
            self.test_qids, self.test_labels, *self.test_data = load_input_tensors(test_jsonl_path, model_type, model_name, max_seq_length, format=format)
            assert all(len(self.test_qids) == x.size(0) for x in [self.test_labels] + self.test_data)
        else:
            raise ValueError('You should either use an official test set or use inhouse split.')

        if self.train_on_subset:
            with open(subset_qids_path, 'r') as fin:
                print(f'loading subset index from {subset_qids_path}')
                subset_qids = set(line.strip() for line in fin)
            self.subset_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in subset_qids])

    def get_node_feature_dim(self):
        return None

    def train_size(self):
        if self.train_on_subset:
            return self.subset_train_indexes.size(0)
        elif self.is_inhouse:
            return self.inhouse_train_indexes.size(0)
        else:
            return len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        elif self.has_official_test:
            return len(self.test_qids)
        else:
            return 0

    def get_train_indexes(self):  # get a list of permuted indices of the training set
        if self.train_on_subset:
            n_train = self.subset_train_indexes.size(0)
            train_indexes = self.subset_train_indexes[torch.randperm(n_train)]
        elif self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            n_train = len(self.train_qids)
            train_indexes = torch.randperm(n_train)
        return train_indexes

    def get_test_indexes(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes
        else:
            return None

    def train(self, train_indexes):
        return BatchGenerator(self.device, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors=self.train_data)

    def dev(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors=self.dev_data)

    def test(self, test_indexes):
        if self.is_inhouse:
            return BatchGenerator(self.device, self.eval_batch_size, test_indexes, self.train_qids, self.train_labels, tensors=self.train_data)
        elif self.has_official_test:
            assert test_indexes is None
            return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors=self.test_data)
        else:
            return None


class TestLMDataLoader(object):
    def __init__(self, test_jsonl_path, eval_batch_size, device, model_name, max_seq_length=128, format=[]):
        super().__init__()
        self.eval_batch_size = eval_batch_size
        self.device = device

        model_type = MODEL_NAME_TO_CLASS[model_name]
        self.test_qids, self.test_labels, *self.test_data = load_input_tensors(test_jsonl_path, model_type, model_name, max_seq_length, format=format)
        assert all(len(self.test_qids) == x.size(0) for x in [self.test_labels] + self.test_data)

    def test_size(self):
        return len(self.test_qids)

    def test(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors=self.test_data)
