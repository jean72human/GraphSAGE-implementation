import torch_geometric

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_scatter import scatter
from torch_geometric.utils import dropout_adj, add_remaining_self_loops


class GraphSAGELayer(torch.nn.Module):
    def __init__(self, sz_in, sz_out, agg="max", dropout=0, depth=1, concat=True, dim_i=1024, normalize=False):
        """
        Args:
            - sz_in: dimention of input
            - sz_out: dimension of output
            - agg: "max" or "mean" aggregator to be used between 
            - edge_index: [2, num_edges] list [(u, v)] for edge from u to v
        """
        super().__init__()
        
        if concat:
            size_one, size_others = 2*sz_in, 2*dim_i
        else:
            size_one, size_others = sz_in, dim_i
        self.concat = concat

        self.depth = depth
        self.linear = nn.ModuleList([nn.Linear(size_one,dim_i, bias=False)] + (depth-1) * [nn.Linear(size_others,sz_out, bias=False)])
        self.sigma = nn.ReLU()
        self.agg = agg
        self.dropout = dropout

        if agg == "max":
            self.agg_model = nn.ModuleList([nn.Linear(sz_in,sz_in, bias=True)] + (depth-1) * [nn.Linear(dim_i,dim_i, bias=True)])

        self.normalize = normalize

    def forward(self, x, edge_index):
        """
        Args:
            - x: [num_nodes, sz_in]
            - edge_index: [2, num_edges] list [(u, v)] for edge from u to v

        Return:
            - new_fts: [N, sz_out]
        """
        N = x.size(0)
        if not self.training:
            edge_index = add_remaining_self_loops(dropout_adj(edge_index, p=self.dropout, force_undirected=True)[0], num_nodes=N)[0]
        u, v = edge_index

        h = x
        for k in range(self.depth):

            if self.agg == "mean": hn = scatter(h[v], u, dim=0, reduce="mean")
            if self.agg == "max": hn = scatter(self.sigma(self.agg_model[k](h[v])), u, dim=0, reduce="max")

            h_cat = torch.cat([h,hn],dim=1) if self.concat else h+hn
            h = self.sigma(self.linear[k](h_cat))
            if self.normalize: h = h/h.norm(dim=1).unsqueeze(-1)

        #assert False, "breakpoint"

        return h