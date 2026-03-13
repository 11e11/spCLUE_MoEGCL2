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
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
    
    def forward(self, z_concat):
        """
        Args:
            z_concat: [N, feature_dim] (Z1和Z2拼接后)
        Returns:
            gate_weights: [N, 2]
        """
        logits = self.gate_net(z_concat)
        bias = torch.tensor([5.0, 0.0]).to(logits.device) 
        gate_weights = F.softmax(logits * 8.0 + bias, dim=1)
        return gate_weights
# class MoEGraphGating(nn.Module):
#     def __init__(self, feature_dim,
#                 hidden_dim=128, dropout_rate=0.1):
#         super().__init__()

#         self.layer1 = nn.Linear(feature_dim, 128)
#         self.dropout1 = nn.Dropout(dropout_rate)

#         self.layer2 = nn.Linear(128, 256)
#         self.leaky_relu1 = nn.LeakyReLU()
#         self.dropout2 = nn.Dropout(dropout_rate)

#         self.layer3 = nn.Linear(256, 128)
#         self.leaky_relu2 = nn.LeakyReLU()
#         self.dropout3 = nn.Dropout(dropout_rate)

#         self.layer4 = nn.Linear(128, 2)

        

#     def forward(self, x):
#         x = torch.relu(self.layer1(x))
#         x = self.dropout1(x)

#         x = self.layer2(x)
#         x = self.leaky_relu1(x)
#         x = self.dropout2(x)

#         x = self.layer3(x)
#         x = self.leaky_relu2(x)
#         x = self.dropout3(x)

#         logits = torch.softmax(self.layer4(x), dim=1)
       
#         return logits


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
        z_concat = F.layer_norm(z_concat, z_concat.size()[1:])   # LayerNorm有助于稳定训练
        gate_weights = self.gating(z_concat)
        gate_weights = 0.7 * gate_weights + 0.3 * torch.spmm(G1, gate_weights)  # [N, 2]
        
        # 根据数据规模选择策略
        if N < self.sparse_threshold:
            # 小数据集: 用密集版本 (模仿MoEGCL)
            return self._dense_fusion(gate_weights, G1, G2)
            # return 0.5*G1+0.5*G2, gate_weights
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
        # Gf = gate_weights[:, 0:1].unsqueeze(1) * G1 + gate_weights[:, 1:2].unsqueeze(1) * G2
        # Gf = 0.3 * G1_dense + 0.7 * G2_dense
        # 变成对称图
        # Gf = (Gf + Gf.T) / 2

        # 归一化处理
        # Gf = self._normalize_adj(Gf)
        
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
    def _normalize_adj(self, adj):
        if adj.is_sparse:
            # 稀疏版归一化
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        
        # 计算度矩阵 D = sum(adj, dim=1)
        D = torch.sum(adj_dense, dim=1) + 1e-8  # 加小值避免除0
        D_sqrt_inv = 1.0 / torch.sqrt(D)
        D_sqrt_inv = torch.diag(D_sqrt_inv)
        
        # 对称归一化：D^-1/2 * adj * D^-1/2
        # adj_norm = torch.matmul(torch.matmul(D_sqrt_inv, adj_dense), D_sqrt_inv)
        adj_norm = torch.matmul(torch.matmul(D_sqrt_inv, adj_dense), D_sqrt_inv)
        
        if adj.is_sparse:
            # 转回稀疏版
            adj_norm_sparse = adj_norm.to_sparse_coo()
            return adj_norm_sparse
        else:
            return adj_norm
    # def _normalize_adj(self, adj):
    #     # 随机游走归一化
    #     if adj.is_sparse:
    #         # 稀疏版：先转稠密计算度（小图），大图可优化为稀疏求和
    #         adj_dense = adj.to_dense()
    #     else:
    #         adj_dense = adj
        
    #     # ========== 关键修改1：计算随机游走归一化的度矩阵 ==========
    #     # D = 每行的和（节点的度），加1e-8避免除0
    #     D = torch.sum(adj_dense, dim=1) + 1e-8  
    #     # 随机游走归一化：仅取D的逆，不做平方根
    #     D_inv = 1.0 / D  # [N,] 一维向量
    #     D_inv = torch.diag(D_inv)  # 转为对角矩阵 [N,N]
        
    #     # ========== 关键修改2：随机游走归一化计算（仅行归一化） ==========
    #     # 公式：A_rw = D^-1 * A （左乘D_inv，仅行归一化）
    #     adj_norm = torch.matmul(D_inv, adj_dense)
        
    #     # ========== 保留稀疏格式返回 ==========
    #     if adj.is_sparse:
    #         # 转回稀疏COO格式，保持与输入一致的存储方式
    #         adj_norm_sparse = adj_norm.to_sparse_coo()
    #         # 合并重复边（稀疏矩阵必备）
    #         adj_norm_sparse = adj_norm_sparse.coalesce()
    #         return adj_norm_sparse
    #     else:
    #         return adj_norm