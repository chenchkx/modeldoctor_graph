
import torch
import argparse, os
import torch.nn as nn
import numpy as np
import random

from utils.utils_second import *

import matplotlib.pyplot as plt
dir_path = os.path.dirname(__file__)


def main(args):

    dataset, train_loader, valid_loader, test_loader = load_data(args)
    args = args_(args, dataset)
    set_seed(args)

    model = load_model(args, dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # load pre-train model
    pth_path = os.path.join(args.dict_dir, args.identity+'.pth')
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))  

    modelOptm = ModelOptLoading(model=model, 
                                optimizer=optimizer,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                test_loader=test_loader,
                                args=args)
    modelOptm.optimizing()
    metric_list = ['train-loss','train-rocauc', 'valid-rocauc', 'test-rocauc']
    print_best_log(args, key_metric='valid-rocauc', eopch_slice=args.epoch_slice)

    plot_logs(args, metric_list)

    print('optmi')

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--datadir", type=str, default='/nfs4-p1/ckx/datasets/ogb/graph/')
    parser.add_argument("--dataset", type=str, default='ogbg-molbbbp')

    parser.add_argument("--model", type=str, default='GCN', choices='GIN, GCN')
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--epoch_slice", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--loss_type", type=str, default='cls', choices='bce, cls')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=0, 
                                help='running times of the second model training')
    
    parser.add_argument("--logs_dir", type=str, default= os.path.join(dir_path,'logs'))

    args = parser.parse_args()
    
    main(args)



