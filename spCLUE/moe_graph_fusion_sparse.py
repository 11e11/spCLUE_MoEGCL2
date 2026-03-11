import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEGraphGating(nn.Module):
    """MoE门控网络"""
    
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, z_concat):
        """
        Args:
            z_concat: [N, feature_dim] (Z1和Z2拼接后)
        Returns:
            gate_weights: [N, 2]
        """
        return self.gate_net(z_concat)


class AdaptiveMoEGraphFusion(nn.Module):
    """
    自适应MoE图融合: 根据数据集大小选择密集或稀疏
    """
    
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1, 
                 sparse_threshold=5000):
        super().__init__()
        self.gating = MoEGraphGating(feature_dim, hidden_dim, dropout)
        self.sparse_threshold = sparse_threshold
    
    def forward(self, z_concat, G1, G2):
        """
        Args:
            z_concat: [N, 2*hidden_dim]
            G1, G2: torch.sparse_coo_tensor 或 torch.Tensor
        
        Returns:
            Gf: 融合图 (自动选择密集或稀疏)
            gate_weights: [N, 2]
        """
        N = z_concat.size(0)
        gate_weights = self.gating(z_concat)
        
        # 根据数据规模选择策略
        if N < self.sparse_threshold:
            # 小数据集: 用密集版本 (模仿MoEGCL)
            return self._dense_fusion(gate_weights, G1, G2)
        else:
            # 大数据集: 用稀疏版本
            return self._sparse_fusion(gate_weights, G1, G2)
    
    def _dense_fusion(self, gate_weights, G1, G2):
        # print("Using dense fusion (MoEGCL style)")
        """密集融合 (MoEGCL的方式)"""
        # 转为密集tensor
        if G1.is_sparse:
            G1_dense = G1.to_dense()
            G2_dense = G2.to_dense()
        else:
            G1_dense = G1
            G2_dense = G2
        
        # MoE融合
        A_list = torch.stack([G1_dense, G2_dense], dim=2)  # [N, N, 2]
        weights = gate_weights.unsqueeze(1)  # [N, 1, 2]
        Gf = torch.sum(A_list * weights, dim=2)  # [N, N]
        
        return Gf, gate_weights
    
    def _sparse_fusion(self, gate_weights, G1_sparse, G2_sparse):
        """稀疏融合 (我们的方式)"""
        # print("Using sparse fusion")
        indices1 = G1_sparse._indices()
        values1 = G1_sparse._values()
        indices2 = G2_sparse._indices()
        values2 = G2_sparse._values()
        
        src_nodes1 = indices1[0]
        src_nodes2 = indices2[0]
        
        w1_edges = gate_weights[src_nodes1, 0]
        w2_edges = gate_weights[src_nodes2, 1]
        
        weighted_values1 = w1_edges * values1
        weighted_values2 = w2_edges * values2
        
        indices_all = torch.cat([indices1, indices2], dim=1)
        values_all = torch.cat([weighted_values1, weighted_values2])
        
        # detach融合图
        Gf_sparse = torch.sparse_coo_tensor(
            indices_all.detach(), 
            values_all.detach(), 
            G1_sparse.size()
        ).coalesce()
        
        return Gf_sparse, gate_weights