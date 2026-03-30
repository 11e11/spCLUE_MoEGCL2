import torch
import torch.nn as nn
import torch.nn.functional as F
from .shared_Encoder import SharedGCNEncoder, SimpleDecoder
from .moe_graph_fusion_sparse import AdaptiveMoEGraphFusion
from .utils import two_hop_propagation


class CCGCN_TwoStage(nn.Module):
    """
    两阶段训练的spCLUE变体
    阶段1: 预训练共享编码器
    阶段2: MoE图融合 + 结构引导对比学习
    """
    
    def __init__(self, dims_list, n_clusters, graph_corr=0.4, dropout=0.5,
                 gate_hidden_dim=128, gate_dropout=0.1, gate_bias=5.0, use_residual=True, residual_weight=0.2):
        super().__init__()
        self.input_dim = dims_list[0]
        self.hidden_dim = dims_list[1]
        self.z_dim = dims_list[2]
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.graph_corr = graph_corr
        self.gate_bias = gate_bias
        self.gate_dropout = gate_dropout
        self.use_residual = use_residual
        self.residual_weight = residual_weight
        
        # === 预训练组件 ===
        self.spatial_encoder = SharedGCNEncoder(
            self.input_dim, self.hidden_dim, dropout, view_specific_norm=False
        )
        self.expr_encoder = SharedGCNEncoder(
            self.input_dim, self.hidden_dim, dropout, view_specific_norm=False
        )
        self.shared_encoder = SharedGCNEncoder(
            self.input_dim, self.hidden_dim, dropout, view_specific_norm=False
        )
        
        # 简单的融合 (预训练阶段用)
        # self.pretrain_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.z_dim) # 降到 64
        )
        
        # 解码器 (预训练和训练阶段共享)
        self.decoder = SimpleDecoder(self.z_dim, self.input_dim)
        
        # === 训练阶段组件 ===
        self.moe_graph_fusion = AdaptiveMoEGraphFusion(
            self.hidden_dim * 2, gate_hidden_dim, gate_dropout, gate_bias
        )
        # === 关键修改1: 视图特定投影头 (用于对比学习) ===
        self.view_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim*2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
        )
        
        # === 关键修改2: 融合表征投影头 (用于对比学习) ===
        self.common_projection = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim*2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
        )
        # 投影头 (原spCLUE保留)
        self.projectInsHead = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
            nn.ReLU(),
        )
        
        self.projectClsHead = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, n_clusters),
            nn.Softmax(dim=1),
        )
        
        self.act = nn.ELU()
        self.gate_stats = {}
    
    def pretrain_forward(self, data, adj1, adj2):
        """
        预训练阶段的前向传播
        
        Returns:
            x_rec: 重构
            z_pretrain: 融合嵌入
        """
        # 在两个图上编码
        z1 = self.shared_encoder(data, adj1, 'spatial', self.graph_corr,self.dropout, True)
        z2 = self.shared_encoder(data, adj2, 'expr', self.graph_corr,self.dropout, True)
        # z1 = self.spatial_encoder(data, adj1, 'spatial', self.graph_corr,self.dropout, False)
        # z2 = self.expr_encoder(data, adj2, 'expr', self.graph_corr,self.dropout, False)
        
        # 简单融合 (可选: 用Attention或直接Concat)
        z_concat = torch.cat([z1, z2], dim=1)
        z_pretrain = self.feature_fusion(z_concat)
        # z_pretrain = z_concat
        # z_pretrain = self.act(self.pretrain_fusion(z_concat))
        
        # 重构
        x_rec = self.decoder(z_pretrain)
        
        return x_rec, z_pretrain
    
    def finetune_forward(self, data, adj1, adj2, freeze_encoder=True):
        """
        训练阶段的前向传播 (图融合 + 对比学习)
        
        Args:
            freeze_encoder: 是否冻结预训练的编码器
        
        Returns:
            z1, z2: 视图特定编码
            z_final: 融合编码 (2-hop传播后)
            x_rec: 重构
            Gf_sparse: 融合图
            gate_weights: 门控权重
        """
        # === Step 1: 用预训练编码器获取Z1, Z2 ===
        if freeze_encoder:
            with torch.no_grad():
                z1 = self.shared_encoder(data, adj1, 'spatial', 0,0, False)  # no dropout
                z2 = self.shared_encoder(data, adj2, 'expr', 0,0, False)
                # z1 = self.spatial_encoder(data, adj1, 'spatial', self.graph_corr,self.dropout, False)
                # z2 = self.expr_encoder(data, adj2, 'expr', self.graph_corr,self.dropout, False)
        else:
            # z1 = self.shared_encoder(data, adj1, 'spatial', self.graph_corr,self.dropout,  False)
            # z2 = self.shared_encoder(data, adj2, 'expr', self.graph_corr, self.dropout, False)
            # MOBV2之前采用的都是如下
            z1 = self.shared_encoder(data, adj1, 'spatial', 0,0.1, False)  
            z2 = self.shared_encoder(data, adj2, 'expr', 0,0.1, False) 

        
        # 1. 投影单视图特征
        h1 = F.normalize(self.view_projection(z1), p=2, dim=1)
        h2 = F.normalize(self.view_projection(z2), p=2, dim=1)

        # z1 = F.normalize(z1, p=2, dim=1)
        # z2 = F.normalize(z2, p=2, dim=1)
        
        # === Step 2: 拼接Z1和Z2 ===
        z_concat = torch.cat([z1, z2], dim=1)  # [N, 2*hidden_dim]
        # z_gate = torch.cat([z_concat, data], dim=1)  # [N, 2*hidden_dim + input_dim]
        
        # === Step 3: MoE图融合 ===
        Gf_sparse, gate_weights = self.moe_graph_fusion(z_concat, adj1, adj2)
        
        # 保存门控统计
        if not self.training:
            w_spatial = gate_weights[:, 0].detach().cpu().numpy()
            w_expr = gate_weights[:, 1].detach().cpu().numpy()
            self.gate_stats = {
                'spatial_mean': float(w_spatial.mean()),
                'spatial_std': float(w_spatial.std()),
                'expr_mean': float(w_expr.mean()),
                'expr_std': float(w_expr.std()),
            }
        
        # === Step 4: 2-hop传播 ===
        z_final = two_hop_propagation(
            z_concat, Gf_sparse, 
            use_residual=self.use_residual, 
            residual_weight=self.residual_weight
        )
        z_final = self.feature_fusion(z_final)  # 降维
        # 降维到hidden_dim (因为z_concat是2*hidden_dim)
        # z_final = self.pretrain_fusion(z_final)
        # Step 6: 融合表征投影 (用于对比学习)
        h_common_proj = F.normalize(self.common_projection(z_final), p=2, dim=1)
        z_final = F.normalize(z_final, p=2, dim=1)
        
        # === Step 5: 重构 ===
        x_rec = self.decoder(z_final)
        
        # return z1, z2, z_final, x_rec, Gf_sparse, gate_weights
        return h1, h2, h_common_proj, z_final, x_rec, Gf_sparse, gate_weights
    # def finetune_forward(self, data, adj1, adj2, freeze_encoder=True):
    #     """训练阶段 (修正版)"""
    #     # === Step 1: 编码 ===
    #     if freeze_encoder:
    #         with torch.no_grad():
    #             z1 = self.shared_encoder(data, adj1, 'spatial', 0, False)
    #             z2 = self.shared_encoder(data, adj2, 'expr', 0, False)
    #     else:
    #         z1 = self.shared_encoder(data, adj1, 'spatial', self.graph_corr, True)
    #         z2 = self.shared_encoder(data, adj2, 'expr', self.graph_corr, True)
        
    #     # === Step 2: 视图特定投影 ===
    #     h1_proj = F.normalize(self.view_projection(z1), p=2, dim=1)
    #     h2_proj = F.normalize(self.view_projection(z2), p=2, dim=1)
        
    #     # === Step 3: 拼接 ===
    #     z_concat = torch.cat([z1, z2], dim=1)  # [N, 128]
        
    #     # === Step 4: MoE图融合 ===
    #     Gf_sparse, gate_weights = self.moe_graph_fusion(z_concat, adj1, adj2)
        
    #     if not self.training:
    #         w_spatial = gate_weights[:, 0].detach().cpu().numpy()
    #         w_expr = gate_weights[:, 1].detach().cpu().numpy()
    #         self.gate_stats = {
    #             'spatial_mean': float(w_spatial.mean()),
    #             'spatial_std': float(w_spatial.std()),
    #             'expr_mean': float(w_expr.mean()),
    #             'expr_std': float(w_expr.std()),
    #         }
        
    #     # === Step 5: 2-hop传播 ===
    #     z_propagated = two_hop_propagation(
    #         z_concat, Gf_sparse, 
    #         use_residual=self.use_residual, 
    #         residual_weight=0.2
    #     )  # [N, 128]
        
    #     # === Step 6: 降维融合 ===
    #     z_fused = self.pretrain_fusion(z_propagated)  # [N, 128] -> [N, 64]
        
    #     # === Step 7: 融合表征投影 ===
    #     h_common_proj = F.normalize(self.common_projection(z_fused), p=2, dim=1)  # [N, 64]
        
    #     # === Step 8: 最终嵌入 ===
    #     z_final = F.normalize(z_fused, p=2, dim=1)  # [N, 64]
        
    #     # === Step 9: 重构 ===
    #     x_rec = self.decoder(z_fused)  # [N, 64] -> [N, 200]
        
    #     return h1_proj, h2_proj, h_common_proj, z_final, x_rec, Gf_sparse, gate_weights
    
    def getCluster(self, embed):
        labels = self.projectClsHead(embed)
        return torch.argmax(labels, dim=1)
    
    def forward(self, data, adj1, adj2, stage='pretrain', freeze_encoder=True):
        """
        统一的前向传播接口
        
        Args:
            stage: 'pretrain' or 'finetune'
        """
        if stage == 'pretrain':
            return self.pretrain_forward(data, adj1, adj2)
        else:
            return self.finetune_forward(data, adj1, adj2, freeze_encoder)