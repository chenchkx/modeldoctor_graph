
import dgl
import dgl.function as fn
import torch

g = dgl.graph(([0, 1, 0, 2, 1, 2, 3, 4, 3, 5], 
               [1, 0, 2, 0, 2, 1, 4, 3, 5, 3]))
g.ndata['x'] = torch.FloatTensor([[1,2],
                                  [1,1],
                                  [1,1],
                                  [1,1],
                                  [1,1],
                                  [1,1]])

g.edata['w'] = torch.FloatTensor([[1,6],
                                  [1,1],
                                  [1,1],
                                  [1,1],
                                  [1,1],
                                  [1,1],
                                  [1,1],
                                  [1,1],
                                  [1,1],
                                  [1,1]])   # each edge has feature size 1

# 对于节点特征，将邻接节点的特征进行aggregate
# update_all 最终的目的是更新到Node级别的特征
g.update_all(fn.copy_u('x', 'm'), fn.sum('m', 'h'))
print(g.ndata['h'])

# 对于边特征，将与Node相连接的边上的特征进行aggregate
g.update_all(fn.copy_e('w', 'e'), fn.sum('e', 'feat'))
print(g.ndata['feat'])

