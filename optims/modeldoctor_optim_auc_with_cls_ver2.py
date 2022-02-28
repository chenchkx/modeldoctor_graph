
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import metrics 
from ogb.graphproppred import Evaluator
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import LambdaLR

criterion = nn.CrossEntropyLoss()

class LinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, t_total, warmup_steps=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(LinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


# class of model optimizing & learning (ModelOptLearning)
class ModelDoctorOptLearning_CLS:
    def __init__(self, model, optimizer, 
                train_loader, valid_loader, test_loader,
                args):
        # initizing ModelOptLearning class
        self.model = model
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        self.args = args
        self.channel_mask = (torch.from_numpy(np.load(args.channel_mask_path)).to(args.device))

    def log_epoch(self, logs_table, train_rst, valid_rst, test_rst, log_lr):
        table_head = []
        table_data = []
        for keys in train_rst.keys():
            table_head.append(f'train-{keys}')
            table_data.append(train_rst[keys])
        for keys in valid_rst.keys():
            table_head.append(f'valid-{keys}')
            table_data.append(valid_rst[keys])
        for keys in test_rst.keys():
            table_head.append(f'test-{keys}')
            table_data.append(test_rst[keys])
        for keys in log_lr.keys():
            table_head.append(f'{keys}')
            table_data.append(log_lr[keys])
        
        return logs_table.append(pd.DataFrame([table_data], columns=table_head), ignore_index=True)

    def eval(self, model, loader, args, channel_mask):
        model.eval()
        total, total_loss = 0, 0  
        y_true, y_soft, y_pred = [], [], []

        for graphs, labels in loader:    
            graphs, labels = graphs.to(args.device), labels.to(args.device)
            nfeats = graphs.ndata['feat']
            efeats = graphs.edata['feat']
            with torch.no_grad():
                outputs = model(graphs, nfeats, efeats, channel_mask)
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
        
    def optimizing(self):
        scheduler = LinearSchedule(self.optimizer, self.args.epochs)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=80, gamma=0.6)
        valid_best = 0
        logs_table = pd.DataFrame()

        print(f'\033[1;32m the performance of label0 channel retained. \033[0m')
        channel_mask = self.channel_mask[0,:]
        self.eval(self.model, self.train_loader, self.args, channel_mask)
        self.eval(self.model, self.valid_loader, self.args, channel_mask)
        self.eval(self.model, self.test_loader, self.args, channel_mask) 

        print(f'\033[1;32m the performance of label1 channel retained. \033[0m')
        channel_mask = self.channel_mask[1,:]
        self.eval(self.model, self.train_loader, self.args, channel_mask)
        self.eval(self.model, self.valid_loader, self.args, channel_mask)
        self.eval(self.model, self.test_loader, self.args, channel_mask) 

        print(f'\033[1;32m the performance of label0 and label1 channel retained. \033[0m')
        channel_mask = torch.sum(self.channel_mask, dim=0)
        channel_mask[channel_mask >=1] = 1     
        self.eval(self.model, self.train_loader, self.args, channel_mask)
        self.eval(self.model, self.valid_loader, self.args, channel_mask)
        self.eval(self.model, self.test_loader, self.args, channel_mask)   


        ####### 
        print(f'\033[1;32m the performance with label0 except label1. \033[0m')
        channel_mask = self.channel_mask[0,:] - self.channel_mask[1,:]
        channel_mask[channel_mask >=1] = 1 
        self.eval(self.model, self.train_loader, self.args, channel_mask)
        self.eval(self.model, self.valid_loader, self.args, channel_mask)
        self.eval(self.model, self.test_loader, self.args, channel_mask) 

        print(f'\033[1;32m the performance with label1 except label0. \033[0m')
        channel_mask = self.channel_mask[1,:] - self.channel_mask[0,:]
        channel_mask[channel_mask >=1] = 1 
        self.eval(self.model, self.train_loader, self.args, channel_mask)
        self.eval(self.model, self.valid_loader, self.args, channel_mask)
        self.eval(self.model, self.test_loader, self.args, channel_mask) 

        print(f'\033[1;32m the performance with the intersection of label0, label1. \033[0m')
        channel_mask = torch.sum(self.channel_mask, dim=0)
        channel_mask[channel_mask <2] = 0  
        channel_mask[channel_mask ==2] = 1  
        self.eval(self.model, self.train_loader, self.args, channel_mask)
        self.eval(self.model, self.valid_loader, self.args, channel_mask)
        self.eval(self.model, self.test_loader, self.args, channel_mask)  


        print(f'\033[1;32m the performance with the union except intersection. \033[0m')
        channel_mask_u = torch.sum(self.channel_mask, dim=0)
        channel_mask_u[channel_mask_u >=1] = 1  
        channel_mask_i = torch.sum(self.channel_mask, dim=0)
        channel_mask_i[channel_mask_i <2] = 0  
        channel_mask_i[channel_mask_i ==2] = 1  
        channel_mask = channel_mask_u - channel_mask_i
        self.eval(self.model, self.train_loader, self.args, channel_mask)
        self.eval(self.model, self.valid_loader, self.args, channel_mask)
        self.eval(self.model, self.test_loader, self.args, channel_mask)  

        # if self.args.channel_retain_class == '0':
        #     channel_mask = self.channel_mask[0,:]
        # elif self.args.channel_retain_class == '1':
        #     channel_mask = self.channel_mask[1,:]
        # else:
        #     channel_mask = torch.sum(self.channel_mask, dim=0)
        #     channel_mask[channel_mask >=1] = 1

        # train_rst = self.eval(self.model, self.train_loader, self.args, channel_mask)
        # valid_rst = self.eval(self.model, self.valid_loader, self.args, channel_mask)
        # test_rst = self.eval(self.model, self.test_loader, self.args, channel_mask) 


        # training model 
        # self.model.train()
        # for graphs, labels in self.train_loader:
        #     graphs, labels = graphs.to(self.args.device), labels.to(self.args.device)
        #     nfeats = graphs.ndata['feat']
        #     efeats = graphs.edata['feat']

        #     outputs = self.model(graphs, nfeats, efeats)
        #     self.optimizer.zero_grad()
        #     loss_cls = criterion(outputs.to(torch.float32), labels.view(-1,))

        #     is_valid = labels == labels
        #     loss_channel = self.gc.loss_channel(outputs, labels.view(-1,)[is_valid.squeeze(1)], graphs.batch_num_nodes())
        #     loss = loss_cls + 100*loss_channel
        #     loss.backward()
        #     self.optimizer.step()

        # train_loss = train_rst['loss']
        # train_perf = train_rst['rocauc']
        # valid_perf = valid_rst['rocauc']
        # test_perf = test_rst['rocauc']

    
        # if not os.path.exists(self.args.doctor_xlsx_dir):
        #     os.mkdir(self.args.doctor_xlsx_dir)
        # logs_table.to_excel(os.path.join(self.args.doctor_xlsx_dir, self.args.doctor_identity+'.xlsx'))
        

