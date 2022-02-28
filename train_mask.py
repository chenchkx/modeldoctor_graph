
import torch
import argparse, os
import torch.nn as nn
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils_mask import *
from utils.grad_constraint_with_cls import HookModule

dir_path = os.path.dirname(__file__)

criterion = nn.CrossEntropyLoss()

def heat_map(args, model, module, device, train_loader, dataset):
    num_sample = 20
    num_classes = int(dataset.num_classes)
    heat_matrix = np.zeros([num_classes, args.embed_dim, num_sample])
    count_matrix = np.zeros([num_classes], dtype=int)

    for graphs, labels in train_loader:
        graphs = graphs.to(device)
        labels = labels.to(device)
        nfeats = graphs.ndata['feat']
        efeats = graphs.edata['feat']

        out = model(graphs, nfeats, efeats)
        out_, indx_ = out.sort(dim = -1, descending=True)
        out_s = out_.softmax(dim = -1)  
        batch_num_nodes = graphs.batch_num_nodes()
        # module = HookModule(model=model, module=model.conv_layers[3])   
        if graphs.batch_size == 1:
            return 0
        for i in range(graphs.batch_size):
            # if bool(out_s[i][0] > 0.98) & bool(step > 20):
            pred_label = indx_[i][0]
            pred_score = out_[i][0]
            if bool((out_s[i][0] > 0.85) & (pred_label == 1)) | bool((out_s[i][0]>0.9) & (pred_label == 0)):  
            # if bool((out_s[i][0] > 0.5) & (out_s[i][0] < 0.6)):     
                if bool(pred_label == labels[i]) & bool(count_matrix[labels[i]]<num_sample) & bool(np.random.randint(0, 2)):                            
                    grads = module.grads(outputs = pred_score, inputs = module.activations)[sum(batch_num_nodes[0:i]):sum(batch_num_nodes[0:i+1]),:]                    
                    grads = abs(grads.cpu().detach().numpy())
                    heat_matrix[labels[i],:,count_matrix[labels[i]]] += np.mean(grads, axis=0)
                    count_matrix[labels[i]] += 1

    heat_matrix_ = heat_matrix.copy()
    heat_matrix_ = heat_matrix_.mean(axis=2)
    heat_matrix_[np.isnan(heat_matrix_)] = 0 
    heat_mask = np.zeros([num_classes, args.embed_dim])
    for i in range(num_classes):
        if count_matrix[i] == 0:
            heat_matrix_[i,:] = 1
            heat_mask[i,heat_matrix_[i,:] > heat_matrix_[i,:].mean()] = 1
        else:
            heat_matrix_[i,:] = heat_matrix_[i,:]/count_matrix[i]
            heat_matrix_[i,:] = (heat_matrix_[i,:] - min(heat_matrix_[i,:]))/(max(heat_matrix_[i,:])-min(heat_matrix_[i,:]))
            heat_mask[i,heat_matrix_[i,:] > heat_matrix_[i,:].mean()] = 1

    if not os.path.exists(args.mask_dir):
        os.mkdir(args.mask_dir)
    sns.heatmap(heat_mask)
    plt.savefig(os.path.join(args.mask_dir, args.identity) + '_heat_mask.png')
    np.save(os.path.join(args.mask_dir, args.identity) + '_heat_mask.npy', heat_mask)
    plt.close()

    sns.heatmap(heat_matrix_)
    plt.savefig(os.path.join(args.mask_dir, args.identity) + '_heat_matrix.png')
    np.save(os.path.join(args.mask_dir, args.identity) + '_heat_matrix.npy', heat_matrix_)
    plt.close()

    sns.heatmap(heat_matrix[0,:,:].T)
    plt.savefig(os.path.join(args.mask_dir, args.identity) + '_heat_matrix0.png')
    plt.close()
    sns.heatmap(heat_matrix[1,:,:].T)
    plt.savefig(os.path.join(args.mask_dir, args.identity) + '_heat_matrix1.png')
    plt.close()


def heat_map_cls(args, model, module, device, train_loader, dataset):
    num_sample = 20
    num_classes = int(dataset.num_classes)
    heat_matrix = np.zeros([num_classes, args.embed_dim, num_sample])
    count_matrix = np.zeros([num_classes], dtype=int)

    for graphs, labels in train_loader:
        graphs = graphs.to(device)
        labels = labels.to(device)
        nfeats = graphs.ndata['feat']
        efeats = graphs.edata['feat']

        out = model(graphs, nfeats, efeats)
        out_, indx_ = out.sort(dim = -1, descending=True)
        out_s = out_.softmax(dim = -1)  
        batch_num_nodes = graphs.batch_num_nodes()
        # module = HookModule(model=model, module=model.conv_layers[3])   
        cls_loss = criterion(out.to(torch.float32), labels.view(-1,))
        batch_grads = module.grads(outputs=cls_loss, inputs=module.activations)

        if graphs.batch_size == 1:
            return 0
        for i in range(graphs.batch_size):
            # if bool(out_s[i][0] > 0.98) & bool(step > 20):
            pred_label = indx_[i][0]
            pred_score = out_[i][0]
            if bool((out_s[i][0] > 0.95) & (pred_label == 1)) | bool((out_s[i][0]>0.85) & (pred_label == 0)):  
            # if bool((out_s[i][0] > 0.5) & (out_s[i][0] < 0.6)):     
                if bool(pred_label == labels[i]) & bool(count_matrix[labels[i]]<num_sample) & bool(np.random.randint(0, 2)):                            
                    grads = batch_grads[sum(batch_num_nodes[0:i]):sum(batch_num_nodes[0:i+1]),:]
                    grads = abs(grads.cpu().detach().numpy())
                    heat_matrix[labels[i],:,count_matrix[labels[i]]] += np.mean(grads, axis=0)
                    count_matrix[labels[i]] += 1

    heat_matrix_ = heat_matrix.copy()
    heat_matrix_ = heat_matrix_.mean(axis=2)
    heat_matrix_[np.isnan(heat_matrix_)] = 0 
    heat_mask = np.zeros([num_classes, args.embed_dim])
    for i in range(num_classes):
        if count_matrix[i] == 0:
            heat_matrix_[i,:] = 1
            heat_mask[i,heat_matrix_[i,:] > heat_matrix_[i,:].mean()] = 1
        else:
            heat_matrix_[i,:] = heat_matrix_[i,:]/count_matrix[i]
            heat_matrix_[i,:] = (heat_matrix_[i,:] - min(heat_matrix_[i,:]))/(max(heat_matrix_[i,:])-min(heat_matrix_[i,:]))
            heat_mask[i,heat_matrix_[i,:] > heat_matrix_[i,:].mean()] = 1

    if not os.path.exists(args.mask_cls_dir):
        os.mkdir(args.mask_cls_dir)
    sns.heatmap(heat_mask)
    plt.savefig(os.path.join(args.mask_cls_dir, args.identity) + '_heat_mask.png')
    np.save(os.path.join(args.mask_cls_dir, args.identity) + '_heat_mask.npy', heat_mask)
    plt.close()

    sns.heatmap(heat_matrix_)
    plt.savefig(os.path.join(args.mask_cls_dir, args.identity) + '_heat_matrix.png')
    np.save(os.path.join(args.mask_cls_dir, args.identity) + '_heat_matrix.npy', heat_matrix_)
    plt.close()

    sns.heatmap(heat_matrix[0,:,:].T)
    plt.savefig(os.path.join(args.mask_cls_dir, args.identity) + '_heat_matrix0.png')
    plt.close()
    sns.heatmap(heat_matrix[1,:,:].T)
    plt.savefig(os.path.join(args.mask_cls_dir, args.identity) + '_heat_matrix1.png')
    plt.close()

def main(args):
    dataset, train_loader, valid_loader, test_loader = load_data(args)
    torch.cuda.set_device(0)
    args = args_(args, dataset)
    set_seed(args)

    model = load_model(args, dataset)
    model.eval()

    module = HookModule(model=model, module=model.conv_layers[3])   
    pth_path = os.path.join(args.dict_dir, args.identity+'.pth')
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.to(args.device)
    heat_map(args, model, module, args.device, train_loader, dataset)
    heat_map_cls(args, model, module, args.device, train_loader, dataset)

    return 0

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--datadir", type=str, default='/nfs4-p1/ckx/datasets/ogb/graph/')
    parser.add_argument("--dataset", type=str, default='ogbg-molbace')

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
    
    parser.add_argument("--logs_dir", type=str, default=os.path.join(dir_path,'logs'))
    parser.add_argument("--mask_dir", type=str, default=os.path.join(dir_path,'mask'))
    parser.add_argument("--mask_cls_dir", type=str, default=os.path.join(dir_path,'mask_cls'))

    args = parser.parse_args()
    
    main(args)



