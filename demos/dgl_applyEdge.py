import dgl
import torch

g = dgl.graph(([0, 1, 0, 2, 1, 2, 3, 4, 3, 5], 
            [1, 0, 2, 0, 2, 1, 4, 3, 5, 3]))
g.ndata['h'] = torch.FloatTensor([[1,2],
                            [1,1],
                            [1,1],
                            [1,1],
                            [1,1],
                            [1,1]])


import dgl.function as fn

# update_all 最终的目的是更新到Node级别的特征
g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_sum'))
print(g.ndata['h_sum'])


g.apply_edges(fn.u_add_v('h', 'h', 'x'))
print(g.edata['x'])

g.apply_edges(fn.copy_u('h', 'e'))
print(g.edata['e'])

