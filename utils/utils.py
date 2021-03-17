import json
import os
import time
import argparse

import torch


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def check_file(file):
    return os.path.isfile(file)


def export_config(config, path):
    param_dict = dict(vars(config))
    check_path(path)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout, indent=4)


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True


def test_data_loader_ms_per_batch(data_loader, max_steps=10000):
    start = time.time()
    n_batch = sum(1 for batch, _ in zip(data_loader, range(max_steps)))
    return (time.time() - start) * 1000 / n_batch


def batch_slice_mask(data, start_idx, end_idx):
    batch_size, dim = data.size()
    idx_expanded = torch.arange(dim).unsqueeze(0).expand(batch_size, -1)
    start_idx_expanded = start_idx.unsqueeze(1).expand(-1, dim)
    end_idx_expanded = end_idx.unsqueeze(1).expand(-1, dim)
    start_idx_mask = idx_expanded >= start_idx_expanded
    end_idx_mask = idx_expanded < end_idx_expanded
    return start_idx_mask & end_idx_mask


def test_batch_slice_mask():
    data = torch.LongTensor([
        [3, 5, 7, 4, 3, 8, 9],
        [1, 4, 7, 8, 9, 1, 2],
        [0, 0, 4, 9, 2, 5, 3]
    ])
    start_idx = torch.LongTensor([1, 2, 3])
    end_idx = torch.LongTensor([2, 5, 1])
    mask = batch_slice_mask(data, start_idx, end_idx)
    print(mask)
