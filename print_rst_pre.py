

import torch
import argparse, os
import torch.nn as nn
import numpy as np
import random
from sklearn import metrics 
from sklearn.metrics import classification_report
from utils.utils import *

import matplotlib.pyplot as plt
dir_path = os.path.dirname(__file__)


def eval(model, loader, args):
    model.eval()
    total, total_loss = 0, 0  
    y_true, y_soft, y_pred = [], [], []

    for graphs, labels in loader:    
        graphs, labels = graphs.to(args.device), labels.to(args.device)
        nfeats = graphs.ndata['feat']
        efeats = graphs.edata['feat']
        with torch.no_grad():
            outputs = model(graphs, nfeats, efeats)
        # y_true.append(labels.view(outputs.shape).detach().cpu())
        y_soft.append(outputs.detach().cpu())
        y_true.append(labels.view(-1, 1).detach().cpu())
        y_pred.append(torch.argmax(outputs.detach(), dim=1).view(-1, 1).cpu())
        total += len(labels)    
        # is_valid = labels == labels
        # loss = criterion(outputs.to(torch.float32)[is_valid], labels.to(torch.float32)[is_valid])
        
    y_true = torch.cat(y_true, dim=0).numpy()
    y_soft = torch.softmax(torch.tensor(torch.cat(y_soft, dim=0).numpy()), dim=1)
    y_pred = torch.cat(y_pred, dim=0).numpy()
    # eval results 
    print(metrics.classification_report(y_true, y_pred, target_names=['class0', 'class1']))
    rst = {}
    rst['rocauc'] = metrics.roc_auc_score(y_true, y_soft[:,1])
    print(f"rocauc: {rst['rocauc']}")

    return rst


def main(args):

    dataset, train_loader, valid_loader, test_loader = load_data(args)
    args = args_(args, dataset)
    set_seed(args)

    model = load_model(args, dataset)
    pth_path = os.path.join(args.dict_dir, args.identity+'.pth')
    model.load_state_dict(torch.load(pth_path))

    metric_list = ['train-loss','train-rocauc', 'valid-rocauc', 'test-rocauc']
    # print_best_log(args, key_metric='valid-rocauc', eopch_slice=args.epoch_slice)
    # plot_logs(args, metric_list)

    print('the performance of train set')
    eval(model, train_loader, args)
    print('the performance of valid set')
    eval(model, valid_loader, args)
    print('the performance of test set')
    eval(model, test_loader, args)

    print('optmi')


if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--datadir", type=str, default='/nfs4-p1/ckx/datasets/ogb/graph/')
    parser.add_argument("--dataset", type=str, default='ogbg-molbbbp')

    parser.add_argument("--model", type=str, default='GCN', choices='GIN, GCN')
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--epoch_slice", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--loss_type", type=str, default='cls', choices='bce, cls')
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--logs_dir", type=str, default= os.path.join(dir_path,'logs'))

    args = parser.parse_args()
    
    main(args)