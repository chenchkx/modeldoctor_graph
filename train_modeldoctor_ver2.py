
import torch
import argparse, os
import torch.nn as nn
import numpy as np
import random
from sklearn import metrics 
from utils.utils_doctor_ver2 import *
# from utils.grad_constraint_with_cls import GradConstraint_CLS
# from utils.grad_constraint_with_clsplus import GradConstraint_CLS

import matplotlib.pyplot as plt
dir_path = os.path.dirname(__file__)
import warnings
warnings.filterwarnings("ignore")

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
    # load pre-train model
    pth_path = os.path.join(args.dict_dir, args.identity+'.pth')
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ### the statistic results of model
    print(f'\033[1;32m the performance of pretain model. \033[0m')
    eval(model, train_loader, args)
    eval(model, valid_loader, args)
    eval(model, test_loader, args)

    # model doctor module
    # modules = [model.conv_layers[layer] for layer in [3]]
    # channel_paths = []
    # channel_paths.append(os.path.join(args.mask_dir, args.identity) + '_heat_mask.npy')
    # channel_paths.append(os.path.join(args.mask_dir, args.identity) + '_heat_matrix.npy')
    # gc = GradConstraint_CLS(model=model, modules=modules, channel_paths=channel_paths, device=args.device)

    ### 
    args.channel_mask_path = os.path.join(args.mask_dir, args.identity) + '_heat_mask.npy'
    args.channel_retain_class = 'all' # '0', '1', 'all'
    modelOptm = ModelOptLoading(model=model, 
                                optimizer=optimizer,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                test_loader=test_loader,
                                args=args)
    modelOptm.optimizing()

    print('optmi')

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--datadir", type=str, default='/nfs4-p1/ckx/datasets/ogb/graph/')
    parser.add_argument("--dataset", type=str, default='ogbg-molbace')

    parser.add_argument("--model", type=str, default='GCN', choices='GIN, GCN')
    parser.add_argument("--epochs", type=int, default=1000)
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
                                help='running times of model doctor')
    
    parser.add_argument("--logs_dir", type=str, default= os.path.join(dir_path,'logs'))
    parser.add_argument("--mask_dir", type=str, default=os.path.join(dir_path,'mask'))

    args = parser.parse_args()
    
    main(args)

