import random
import time
import sys
import logging
import csv
import socket
from shutil import copyfile
from datetime import datetime

from transformers import get_constant_schedule, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils.optimization_utils import OPTIMIZER_CLASSES
from utils.utils import unfreeze_net, check_path
from utils.parser_utils import get_parser, bool_flag
from modeling.modeling_hgn import *


def evaluate_accuracy(graph_loader, eval_set, model, device):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for graph, (qids, labels, *lm_input_data) in zip(graph_loader, eval_set):
            graph = graph.to(device)
            edge_index = graph.edge_index
            node_batch = graph.batch
            num_of_nodes = graph.num_of_nodes
            num_of_edges = graph.num_of_edges
            rel_ids_embs = graph.edge_attr
            c_ids = graph.x
            c_types = graph.node_type
            logits, unnormalized_wts, normalized_wts = model(*lm_input_data, edge_index=edge_index, c_ids=c_ids, c_types=c_types, node_batch=node_batch, rel_ids_embs=rel_ids_embs, num_of_nodes=num_of_nodes, num_of_edges=num_of_edges)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'pred'], help='run training or evaluation')
    parser.add_argument('--save_dir', required=True, help='model output directory')
    parser.add_argument('--save_file_name', default='')
    parser.add_argument('--save_model', default=True, type=bool_flag)

    # statements
    parser.add_argument('--train_jsonl', required=True)
    parser.add_argument('--dev_jsonl', required=True)
    parser.add_argument('--test_jsonl')

    # data
    parser.add_argument('--num_choice', type=int, required=True, help='how many choices for each question')

    parser.add_argument('--train_adj_pk', required=True)
    parser.add_argument('--train_gen_pt', required=True)

    parser.add_argument('--dev_adj_pk', required=True)
    parser.add_argument('--dev_gen_pt', required=True)

    parser.add_argument('--test_adj_pk')
    parser.add_argument('--test_gen_pt')

    # pred mode
    parser.add_argument('--test_path_base')
    parser.add_argument('--test_model_path')
    parser.add_argument('--output_pred_path')
    parser.add_argument('--output_graph', default=False, type=bool_flag)

    # model architecture
    parser.add_argument('--ablation', default=[], nargs='+', choices=['GAT', 'no_edge_weight', 'extraction_only', 'unnormalized_edge_weight', 'wo_statement_vec'])
    # no_edge_weight = no learnable edge weight in message passing (all weights = 1) + no sparsity loss
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--mlp_dim', default=128, type=int, help='number of MLP hidden units')
    parser.add_argument('--fc_dim', default=128, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')
    parser.add_argument('--emb_scale', default=1.0, type=float, help='scale pretrained embeddings')
    parser.add_argument('--num_gnn_layers', default=1, type=int, help='scale pretrained embeddings')
    # regularization
    parser.add_argument('--dropoutm', type=float, default=0.3, help='dropout for mlp hidden units (0 = no dropout')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)  # batch size should be divisible by mini batch size in current implementation.
    parser.add_argument('-ebs', '--eval_batch_size', default=-1, type=int)
    parser.add_argument('--unfreeze_epoch', default=0, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--eval_interval', default=0, type=int, help='steps_per_eval (0 = eval after each epoch)')

    # specific to Hybrid GN
    parser.add_argument('--alpha', default=0, type=float, help='weight for binary loss')
    parser.add_argument('--edge_weight_dropout', default=0.2, type=float)

    # CODAH
    parser.add_argument('--warmup_ratio', default=None, type=float)
    parser.add_argument('--use_last_epoch', default=False, type=bool_flag)
    parser.add_argument('--fold', default=None, type=str)

    args = parser.parse_args()
    if args.test_path_base is not None:
        args.test_model_path = args.test_path_base + args.test_model_path
        args.output_pred_path = args.test_path_base + args.output_pred_path
    if True:  # args.eval_batch_size == -1:
        args.eval_batch_size = args.mini_batch_size  # should be the same due to test data loader
    if args.mini_batch_size > args.batch_size:
        args.mini_batch_size = args.batch_size
    if args.batch_size % args.mini_batch_size != 0:
        raise ValueError('batch size should be divisible by mini batch size')
    if args.fold is not None:
        args.train_jsonl = args.train_jsonl.replace('{fold}', args.fold)
        args.dev_jsonl = args.dev_jsonl.replace('{fold}', args.fold)
        args.test_jsonl = args.test_jsonl.replace('{fold}', args.fold)
        args.train_adj_pk = args.train_adj_pk.replace('{fold}', args.fold)
        args.dev_adj_pk = args.dev_adj_pk.replace('{fold}', args.fold)
        args.test_adj_pk = args.test_adj_pk.replace('{fold}', args.fold)
        args.train_gen_pt = args.train_gen_pt.replace('{fold}', args.fold)
        args.dev_gen_pt = args.dev_gen_pt.replace('{fold}', args.fold)
        args.test_gen_pt = args.test_gen_pt.replace('{fold}', args.fold)
        args.save_dir = args.save_dir.replace('{fold}', args.fold)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    unique_str = datetime.now().strftime("%m%d_%H%M%S.%f") + args.save_file_name
    log_name = unique_str + '.log'
    log_path = os.path.join(args.save_dir, log_name)
    check_path(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    args.save_file_name = unique_str + '.pt'
    if args.mode == 'train':
        dev_acc, test_acc, best_test_acc = train(args)
        new_file_str = f'dlr{args.decoder_lr}_{dev_acc * 100:.2f}_{test_acc * 100:.2f}_s{args.seed}_{best_test_acc * 100:.2f}_{unique_str}'
        os.rename(log_path, os.path.join(args.save_dir, new_file_str + '.log'))
        if args.save_model:
            os.rename(os.path.join(args.save_dir, unique_str + '.pt'), os.path.join(args.save_dir, new_file_str + '.pt'))
    elif args.mode == 'eval':
        eval(args)
    elif args.mode == 'pred':
        pred(args)
    else:
        raise ValueError('Invalid mode')


def train(args):
    logging.info(f'{socket.gethostname()}: {os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else "unknown"}')
    logging.info('python ' + ' '.join(sys.argv))
    logging.info(args)

    model_path = os.path.join(args.save_dir, args.save_file_name)
    check_path(model_path)

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################

    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1))
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)

    rel_emb = np.load(args.rel_emb_path)
    rel_emb = np.concatenate((rel_emb, -rel_emb), 0)
    rel_emb = torch.tensor(rel_emb)
    relation_num, relation_dim = rel_emb.size(0), rel_emb.size(1)
    logging.info('| num_concepts: {} | num_relations: {} |'.format(concept_num, relation_num))

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    lm_data_loader = LMDataLoader(args.train_jsonl, args.dev_jsonl, args.test_jsonl,
                                  batch_size=args.mini_batch_size, eval_batch_size=args.eval_batch_size, device=device,
                                  model_name=args.encoder, max_seq_length=args.max_seq_len, is_inhouse=args.inhouse,
                                  inhouse_train_qids_path=args.inhouse_train_qids, subset_qids_path=args.subset_train_qids,
                                  format=args.format)
    logging.info(f'| # train questions: {lm_data_loader.train_size()} | # dev questions: {lm_data_loader.dev_size()} | # test questions: {lm_data_loader.test_size()} |')

    ###################################################################################################
    #   Build model                                                                                   #
    ###################################################################################################
    graph_data_loader = GraphDataLoader(args.train_adj_pk, args.train_gen_pt, args.dev_adj_pk, args.dev_gen_pt,
                                        args.test_adj_pk, args.test_gen_pt,
                                        args.mini_batch_size, args.eval_batch_size, args.num_choice, args.ablation)
    train_avg_node_num, train_avg_edge_num = graph_data_loader.get_pyg_loader(lm_data_loader.get_train_indexes(), stats_only=True)

    dev_lm_data_loader = lm_data_loader.dev()
    dev_graph_loader, dev_avg_node_num, dev_avg_edge_num = graph_data_loader.dev_graph_data()
    assert len(dev_graph_loader) == len(dev_lm_data_loader)

    if args.inhouse:
        test_index = lm_data_loader.get_test_indexes()
        test_graph_loader, test_avg_node_num, test_avg_edge_num = graph_data_loader.get_pyg_loader(test_index)
    else:
        test_index = None
        test_graph_loader, test_avg_node_num, test_avg_edge_num = graph_data_loader.test_graph_data()
    test_lm_data_loader = lm_data_loader.test(test_index)
    assert len(test_graph_loader) == len(test_lm_data_loader)

    logging.info(f'| train | avg node num: {train_avg_node_num:.2f} | avg edge num: {train_avg_edge_num:.2f} |')
    logging.info(f'| dev   | avg node num: {dev_avg_node_num:.2f} | avg edge num: {dev_avg_edge_num:.2f} |')
    logging.info(f'| test  | avg node num: {test_avg_node_num:.2f} | avg edge num: {test_avg_edge_num:.2f} |')

    model = LMGraphNet(model_name=args.encoder, encoder_pooler=args.encoder_pooler,
                       concept_num=concept_num, concept_dim=relation_dim,
                       relation_num=relation_num, relation_dim=relation_dim, concept_in_dim=concept_dim,
                       hidden_size=args.mlp_dim, num_attention_heads=args.att_head_num,
                       fc_size=args.fc_dim, num_fc_layers=args.fc_layer_num, dropout=args.dropoutm,
                       edge_weight_dropout=args.edge_weight_dropout,
                       pretrained_concept_emb=cp_emb,  pretrained_relation_emb=rel_emb,
                       freeze_ent_emb=args.freeze_ent_emb, num_layers=args.num_gnn_layers,
                       ablation=args.ablation, emb_scale=args.emb_scale,
                       aristo_path=args.aristo_path)

    model.to(device)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    if args.lr_schedule == 'fixed':
        scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (lm_data_loader.train_size() / args.batch_size))
        if args.warmup_ratio is not None:
            warmup_steps = int(args.warmup_ratio * max_steps)
        else:
            warmup_steps = args.warmup_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

    logging.info('parameters:')
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            logging.info('\t{:45}\ttrainable\t{}'.format(name, param.size()))
        else:
            logging.info('\t{:45}\tfixed\t{}'.format(name, param.size()))
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    logging.info(f'\ttotal: {num_params}')

    loss_func = nn.CrossEntropyLoss(reduction='mean')

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    logging.info('')
    logging.info('-' * 71)
    global_step, eval_id, best_dev_id, best_dev_step = 0, 0, 0, 0
    best_dev_acc, final_test_acc, best_test_acc, total_loss = 0.0, 0.0, 0.0, 0.0
    best_test_acc = 0.0
    exit_training = False
    train_start_time = time.time()
    start_time = train_start_time
    model.train()
    freeze_net(model.encoder)
    try:
        binary_score_lst = []
        for epoch_id in range(args.n_epochs):
            if exit_training:
                break
            if epoch_id == args.unfreeze_epoch:
                logging.info('encoder unfreezed')
                unfreeze_net(model.encoder)
            if epoch_id == args.refreeze_epoch:
                logging.info('encoder refreezed')
                freeze_net(model.encoder)
            model.train()
            i = 0
            optimizer.zero_grad()
            train_index = lm_data_loader.get_train_indexes()
            train_graph_loader, train_avg_node_num, train_avg_edge_num = graph_data_loader.get_pyg_loader(train_index)
            train_lm_data_loader = lm_data_loader.train(train_index)
            assert len(train_graph_loader) == len(train_lm_data_loader)
            for graph, (qids, labels, *lm_input_data) in zip(train_graph_loader, train_lm_data_loader):
                graph = graph.to(device)
                edge_index = graph.edge_index
                row, col = edge_index
                node_batch = graph.batch
                num_of_nodes = graph.num_of_nodes
                num_of_edges = graph.num_of_edges
                rel_ids_embs = graph.edge_attr
                c_ids = graph.x
                c_types = graph.node_type
                logits, unnormalized_wts, normalized_wts = model(*lm_input_data, edge_index=edge_index, c_ids=c_ids, c_types=c_types, node_batch=node_batch, rel_ids_embs=rel_ids_embs, num_of_nodes=num_of_nodes, num_of_edges=num_of_edges)
                loss = loss_func(logits, labels)  # scale: loss per question
                if 'no_edge_weight' not in args.ablation and 'GAT' not in args.ablation:  # add options for other kinds of sparsity
                    log_wts = torch.log(normalized_wts + 0.0000001)
                    entropy = - normalized_wts * log_wts  # entropy: [num_of_edges in the batched graph, 1]
                    entropy = scatter_mean(entropy, node_batch[row], dim=0, dim_size=args.mini_batch_size * args.num_choice)
                    loss += args.alpha * torch.mean(entropy)  # scale: entropy per graph (each question has num_choice graphs)
                loss = loss * args.mini_batch_size / args.batch_size  # will be accumulated for (args.batch_size / args.mini_batch_size) times
                loss.backward()
                total_loss += loss.item()
                if 'no_edge_weight' not in args.ablation and 'GAT' not in args.ablation:
                    binary_score_lst += entropy.squeeze().tolist()
                else:
                    binary_score_lst.append(0)
                i = i + args.mini_batch_size
                if i % args.batch_size == 0:
                    if args.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()  # bp: scale: loss per question
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if global_step % args.log_interval == 0:
                        total_loss /= args.log_interval
                        ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                        logging.info('| step {:5} | lr: {:9.7f} | loss {:7.20f} | entropy score {:7.4f} | ms/batch {:7.2f} |'
                                     .format(global_step, scheduler.get_lr()[0], total_loss, np.mean(binary_score_lst), ms_per_batch))
                        total_loss = 0
                        binary_score_lst = []
                        start_time = time.time()
                    if args.eval_interval > 0:
                        if global_step % args.eval_interval == 0:
                            eval_id += 1
                            model.eval()
                            dev_acc = evaluate_accuracy(dev_graph_loader, dev_lm_data_loader, model, device)
                            test_acc = evaluate_accuracy(test_graph_loader, test_lm_data_loader, model, device)
                            # test_acc = 0.2
                            best_test_acc = max(best_test_acc, test_acc)
                            logging.info('-' * 71)
                            logging.info('| step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(global_step, dev_acc, test_acc))
                            logging.info('-' * 71)
                            if dev_acc >= best_dev_acc:
                                best_dev_acc = dev_acc
                                final_test_acc = test_acc
                                best_dev_id = eval_id
                                best_dev_step = global_step
                                if args.save_model:
                                    torch.save(model.state_dict(), model_path)
                                    copyfile(model_path, f'{model_path}_{global_step}_{dev_acc*100:.2f}_{test_acc*100:.2f}.pt')  # tmp
                                logging.info(f'model saved to {model_path}')
                            else:
                                logging.info(f'hit patience {eval_id - best_dev_id}/{args.patience}')
                            model.train()
                            if epoch_id > args.unfreeze_epoch and eval_id - best_dev_id >= args.patience:
                                exit_training = True
                                break
            if args.eval_interval == 0:
                eval_id += 1
                model.eval()
                dev_acc = evaluate_accuracy(dev_graph_loader, dev_lm_data_loader, model, device)
                test_acc = evaluate_accuracy(test_graph_loader, test_lm_data_loader, model, device)
                best_test_acc = max(best_test_acc, test_acc)
                logging.info('-' * 71)
                logging.info('| epoch {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, dev_acc, test_acc))
                logging.info('-' * 71)
                if dev_acc >= best_dev_acc:
                    best_dev_acc = dev_acc
                    final_test_acc = test_acc
                    best_dev_id = eval_id
                    best_dev_step = global_step
                    if args.save_model:
                        torch.save(model.state_dict(), model_path)
                    logging.info(f'model saved to {model_path}')
                else:
                    logging.info(f'hit patience {eval_id - best_dev_id}/{args.patience}')
                model.train()
                if epoch_id > args.unfreeze_epoch and eval_id - best_dev_id >= args.patience:
                    exit_training = True
                    break
            start_time = time.time()
    except KeyboardInterrupt:
        logging.info('-' * 89)
        logging.info('Exiting from training early')
    train_end_time = time.time()
    logging.info('')
    logging.info(f'training ends in {global_step} steps, {train_end_time - train_start_time:.0f} s')
    logging.info('best dev acc: {:.4f} (at step {})'.format(best_dev_acc, best_dev_step))
    logging.info('final test acc: {:.4f}'.format(final_test_acc))
    if args.use_last_epoch:
        logging.info(f'last dev acc: {dev_acc:.4f}')
        logging.info(f'last test acc: {test_acc:.4f}')
        return dev_acc, test_acc, best_test_acc
    else:
        return best_dev_acc, final_test_acc, best_test_acc


def pred(args):
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1))
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)

    rel_emb = np.load(args.rel_emb_path)
    rel_emb = np.concatenate((rel_emb, -rel_emb), 0)
    rel_emb = torch.tensor(rel_emb)
    relation_num, relation_dim = rel_emb.size(0), rel_emb.size(1)
    logging.info('| num_concepts: {} | num_relations: {} |'.format(concept_num, relation_num))

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    lm_data_loader = TestLMDataLoader(args.test_jsonl, eval_batch_size=args.eval_batch_size, device=device,
                                      model_name=args.encoder, max_seq_length=args.max_seq_len, format=args.format)
    logging.info(f'| test size: {lm_data_loader.test_size()} |')

    graph_data_loader = TestGraphDataLoader(args.test_adj_pk, args.test_gen_pt, args.eval_batch_size, args.num_choice, args.ablation)
    test_graph_loader, test_avg_node_num, test_avg_edge_num = graph_data_loader.test_graph_data()

    model = LMGraphNet(model_name=args.encoder, encoder_pooler=args.encoder_pooler,
                       concept_num=concept_num, concept_dim=relation_dim,
                       relation_num=relation_num, relation_dim=relation_dim, concept_in_dim=concept_dim,
                       hidden_size=args.mlp_dim, num_attention_heads=args.att_head_num,
                       fc_size=args.fc_dim, num_fc_layers=args.fc_layer_num, dropout=args.dropoutm,
                       edge_weight_dropout=args.edge_weight_dropout,
                       pretrained_concept_emb=cp_emb, pretrained_relation_emb=rel_emb,
                       freeze_ent_emb=args.freeze_ent_emb, num_layers=args.num_gnn_layers,
                       ablation=args.ablation, emb_scale=args.emb_scale)
    model.to(device)
    model.load_state_dict(torch.load(args.test_model_path))
    model.eval()
    preds = []
    question_ids = []
    all_output_graphs = []
    n_samples, n_correct = 0, 0
    with torch.no_grad():
        test_lm_data_loader = lm_data_loader.test()
        assert len(test_graph_loader) == len(test_lm_data_loader)
        for graph, (qids, labels, *lm_input_data) in zip(tqdm(test_graph_loader, desc='testing'), test_lm_data_loader):
            graph = graph.to(device)
            edge_index = graph.edge_index
            node_batch = graph.batch
            num_of_nodes = graph.num_of_nodes
            num_of_edges = graph.num_of_edges
            rel_ids_embs = graph.edge_attr
            c_ids = graph.x
            c_types = graph.node_type
            logits, unnormalized_wts, normalized_wts = model(*lm_input_data, edge_index=edge_index, c_ids=c_ids, c_types=c_types, node_batch=node_batch, rel_ids_embs=rel_ids_embs, num_of_nodes=num_of_nodes, num_of_edges=num_of_edges)
            if args.output_graph:
                row, col = edge_index
                row = row.cpu().numpy()
                col = col.cpu().numpy()
                edge_batch = node_batch[row].cpu().numpy()
                num_of_nodes_lst = graph.num_of_nodes.tolist()
                index_offset = np.cumsum([0] + num_of_nodes_lst[:-1])
                scattered_index_offset = np.array([index_offset[edge_batch[i]] for i in range(graph.num_edges)])
                unbatched_row = row - scattered_index_offset
                unbatched_col = col - scattered_index_offset
                output_graphs = [[] for _ in range(graph.num_graphs)]
                for src_node, tgt_node, weight, graph_id in zip(unbatched_row, unbatched_col, normalized_wts.squeeze().cpu().numpy(), edge_batch):
                    src_local_id = src_node.item()
                    tgt_local_id = tgt_node.item()
                    output_graphs[graph_id.item()].append((src_local_id, tgt_local_id, weight.item()))
                unbatched_graph_lst = graph.to_data_list()
                for graph_id in range(graph.num_graphs):
                    local_x = unbatched_graph_lst[graph_id].x.tolist()
                    local_rel_id = unbatched_graph_lst[graph_id].edge_attr[:, 0].int().tolist()
                    for edge_id in range(len(output_graphs[graph_id])):
                        src_local_id, tgt_local_id, weight = output_graphs[graph_id][edge_id]
                        src_global_id = local_x[src_local_id]
                        tgt_global_id = local_x[tgt_local_id]
                        rel_id = local_rel_id[edge_id]
                        output_graphs[graph_id][edge_id] = (src_global_id, tgt_global_id, rel_id, weight)
                    output_graphs[graph_id] = sorted(output_graphs[graph_id], key=lambda x: x[-1], reverse=True)
                all_output_graphs += output_graphs
            preds += logits.argmax(1).tolist()
            question_ids += qids
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    assert len(question_ids) == len(preds)
    logging.info(f'Accuracy: {n_correct / n_samples * 100:.2f}')
    with open(args.output_pred_path, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f, delimiter=',')
        for question_id, pred in zip(question_ids, preds):
            csv_writer.writerow([question_id, chr(ord('A') + pred)])
    logging.info(f'Predictions written to {args.output_pred_path}')
    if args.output_graph:
        graph_output_path = args.output_pred_path + '.graph.raw'
        with open(graph_output_path, 'w', encoding='utf-8') as f:
            for graph in all_output_graphs:
                f.write(json.dumps(graph) + '\n')
        logging.info(f'Graphs written to {graph_output_path}')


if __name__ == '__main__':
    main()
