import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseLayer(nn.Module):
    def __init__(self, alpha=0.01, dropout=0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.drop = dropout

    def forward(self, x):
        gauss_x = x + self.alpha * torch.randn_like(x)
        return F.dropout(gauss_x, self.drop, training=self.training)
class TransForm_W(nn.Module):

    def __init__(self, input_dim, out_dim, dropout=0.5, act=None) -> None:
        super().__init__()
        self.dropout = dropout
        self.W = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(input_dim, out_dim))
        )  ## = initialize weight of transform layer

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x @ self.W
class SharedGCNEncoder(nn.Module):
    """
    共享的单层GCN编码器 (用于预训练)
    两个视图共享权重,但可选视图特定归一化
    """
    
    def __init__(self, input_dim, hidden_dim, dropout=0.5, view_specific_norm=True):
        super().__init__()
        self.noiseLayer = NoiseLayer(dropout=dropout)
        self.transform = TransForm_W(input_dim, hidden_dim, dropout)
        self.act = nn.ELU()
        
        # 可选: 视图特定的BatchNorm
        self.view_specific_norm = view_specific_norm
        if view_specific_norm:
            self.bn_spatial = nn.BatchNorm1d(hidden_dim)
            self.bn_expr = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, data, adj, view_type=None, graph_corr=0.4, training=True):
        """
        Args:
            data: [N, input_dim]
            adj: sparse adjacency matrix
            view_type: 'spatial' or 'expr' (用于视图特定归一化)
            graph_corr: edge dropout probability
        """
        feature = self.noiseLayer(data)
        
        # GCN with edge dropout
        adj_dropped = torch.sparse_coo_tensor(
            adj._indices(),
            F.dropout(adj._values(), p=graph_corr, training=training),
            size=adj.size(),
        )
        
        feature = self.act(torch.spmm(adj_dropped, self.transform(feature)))
        
        # 可选: 视图特定归一化
        if self.view_specific_norm and view_type is not None:
            if view_type == 'spatial':
                feature = self.bn_spatial(feature)
            elif view_type == 'expr':
                feature = self.bn_expr(feature)
        
        return feature


class SimpleDecoder(nn.Module):
    """简单的解码器 (用于重构)"""
    
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, z):
        return self.decoder(z)