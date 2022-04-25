import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GraphConv, GCNConv, GATConv, SAGEConv, AGNNConv


class GNNSimple(nn.Module):
    def __init__(self, in_dim, cfg, node_dim=0):
        super(GNNSimple, self).__init__()
        self.cfg = cfg
        self.node_dim = node_dim
        self.hdims = hdims = cfg['hdims']
        self.num_layers = len(hdims)
        self.gconv_layers = nn.ModuleList()
        self.residual = cfg.get('residual', False)
        self.cat_input = cfg.get('cat_input', False)
        self.num_layer_update = cfg.get('num_layer_update', 1)
        self.in_fc = nn.Linear(in_dim, hdims[0])
        self.out_dim = hdims[-1] + (in_dim if self.cat_input else 0)
        act = cfg.get('act', 'relu')
        if act == 'relu':
            self.activation = torch.relu
        elif act == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = torch.sigmoid

        for i in range(self.num_layers):
            conv_kwargs = {
                'in_channels': hdims[0] if i == 0 else hdims[i-1],
                'out_channels': hdims[i],
                'bias': cfg['bias'],
                'node_dim': node_dim
            }
            if cfg['layer_type'] == 'graph_conv':
                conv_kwargs['aggr'] = cfg['aggr']
                gnn_cls = GraphConv
            elif cfg['layer_type'] == 'gcn_conv':
                gnn_cls = GCNConv
            elif cfg['layer_type'] == 'gat_conv':
                gnn_cls = GATConv
                del conv_kwargs['node_dim']
                conv_kwargs['heads'] = cfg.get('heads', 1)
            elif cfg['layer_type'] == 'sage_conv':
                gnn_cls = SAGEConv
            elif cfg['layer_type'] == 'agnn_conv':
                gnn_cls = AGNNConv
                conv_kwargs = {}
            else:
                raise ValueError('unknown gnn layer type!')
            gnn_layer = gnn_cls(**conv_kwargs)
            self.gconv_layers.append(gnn_layer)

    def forward(self, x, edge_index):
        # loop all the layers
        x_init = x
        x = self.in_fc(x)
        for layer_index in range(self.num_layers):
            for _ in range(self.num_layer_update):
                gconv = self.gconv_layers[layer_index]
                # update node features
                x_in = x
                x = gconv(x, edge_index=edge_index)
                x = self.activation(x)
                if self.residual and x.shape[-1] == x_in.shape[-1]:
                    x += x_in
        if self.cat_input:
            x = torch.cat([x, x_init], dim=-1)
        return x