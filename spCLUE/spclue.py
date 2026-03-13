import torch
import numpy as np

from .loss import *
from tqdm import tqdm
from .utils import *
from sklearn.metrics import adjusted_rand_score
from .network import CCGCN_TwoStage


class spCLUE_TwoStage:
    """两阶段训练的spCLUE (修正版 - 包含updateResult)"""
    
    def __init__(
        self,
        input_data,
        graph_dict,
        n_clusters=12,
        pretrain_epochs=100,
        finetune_epochs=100,
        random_seed=0,
        device=torch.device("cuda:0"),
        learning_rate=0.001,
        weight_decay=5e-4,
        dim_input=200,
        dim_hidden=64,
        dim_embed=64,
        graph_corr=0.4,
        dropout=0.5,
        gamma=1.0,
        beta=0.0,
        theta=5.0,
        kappa=1.0,
        gate_hidden_dim=128,
        gate_dropout=0.1,
        gate_bias=5.0,
        freeze_encoder=True,
        residual_weight=0.2
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.gamma = gamma
        self.beta = beta
        self.theta = theta
        self.kappa = kappa
        self.gate_bias = gate_bias
        self.gate_dropout = gate_dropout
        self.dims_list = [dim_input, dim_hidden, dim_embed]
        self.n_spot = input_data.shape[0]
        self.freeze_encoder = freeze_encoder
        self.residual_weight = residual_weight
        
        fix_seed(self.random_seed)
        
        # 数据
        self.input_data = torch.FloatTensor(input_data).to(self.device)
        self.g_spatial = sparse_mx_to_torch_sparse_tensor(graph_dict["spatial"]).to(self.device)
        self.g_expr = sparse_mx_to_torch_sparse_tensor(graph_dict["expr"]).to(self.device)
        
        # 模型
        self.model = CCGCN_TwoStage(
            self.dims_list, self.n_clusters, graph_corr, dropout,
            gate_hidden_dim, gate_dropout,gate_bias, self.residual_weight
        ).to(self.device)
        
        # 损失函数
        self.rec_crit = MSELoss()
        self.contrast_crit = StructureGuidedContrastiveLoss(device=self.device)
        self.cluster_crit = ClusterLoss(self.n_clusters, self.device)
    
    def updateResult(self):
        """
        获取当前模型的聚类结果和嵌入
        
        功能:
        1. 切换到评估模式
        2. 前向传播获取最终嵌入
        3. 聚类并返回结果
        
        Returns:
            predLabel: [N] 聚类标签
            features_fuse: [N, D] 融合嵌入
        """
        with torch.no_grad():
            self.model.eval()
            
            # 前向传播 (finetune模式)
            # z1, z2, z_final, _, Gf_sparse, gate_weights = self.model(
            #     self.input_data, self.g_spatial, self.g_expr,
            #     stage='finetune', freeze_encoder=True,
            # )
            h1_proj, h2_proj, h_common_proj, z_final, x_rec, Gf_sparse, gate_weights = self.model(
                self.input_data, self.g_spatial, self.g_expr,
                stage='finetune', freeze_encoder=True
            )
            
            # 聚类
            predLabel = self.model.getCluster(z_final)
            
            
            # 转为numpy
            features_fuse = z_final.detach().cpu().numpy()
            predLabel = predLabel.detach().cpu().numpy()
            gate_weights = gate_weights.detach().cpu().numpy()
            
            # 打印门控统计 (调试用)
            stats = self.model.gate_stats
            if stats:  # 如果有统计信息
                print(f"  [Gate Stats] spatial={stats['spatial_mean']:.3f}±{stats['spatial_std']:.3f}, "
                      f"expr={stats['expr_mean']:.3f}±{stats['expr_std']:.3f}")
        
        return predLabel, features_fuse, gate_weights
    
    def pretrain(self):
        """阶段1: 预训练共享编码器"""
        print("=" * 60)
        print("Stage 1: Pre-training Shared Encoder")
        print("=" * 60)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        for epoch in tqdm(range(self.pretrain_epochs), desc="Pretrain"):
            self.model.train()
            adjust_learning_rate(optimizer, epoch, self.learning_rate)
            optimizer.zero_grad()
            
            # 前向传播 (pretrain模式)
            x_rec, _ = self.model(
                self.input_data, self.g_spatial, self.g_expr, stage='pretrain'
            )
            
            # 重构损失
            loss = self.rec_crit(x_rec, self.input_data)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Pretrain Epoch {epoch+1}: Rec Loss = {loss.item():.6f}")
        
        print(f"✓ Pretrain finished! Final Rec Loss = {loss.item():.6f}\n")
    
    def finetune(self):
        """阶段2: 训练 (MoE图融合 + 对比学习)"""
        print("=" * 60)
        print("Stage 2: Fine-tuning with MoE Graph Fusion")
        print("=" * 60)
        
        # 设置优化器
        if self.freeze_encoder:
            trainable_params = [
                p for n, p in self.model.named_parameters() 
                if 'shared_encoder' not in n
            ]
            print("✓ Encoder frozen, only training fusion & projection heads")
        else:
            trainable_params = self.model.parameters()
            print("✓ Encoder unfrozen, training all parameters")
        
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 早停阈值 (根据数据集大小调整)
        max_ari = 0.3 if self.n_spot <= 10000 else 1.1
        
        print(f"✓ Early stopping threshold: ARI >= {max_ari:.2f}\n")
        
        for epoch in tqdm(range(self.finetune_epochs), desc="Finetune"):
            self.model.train()
            adjust_learning_rate(optimizer, epoch, self.learning_rate)
            optimizer.zero_grad()
            
            # 前向传播 (finetune模式)
            # z1, z2, z_final, x_rec, Gf_sparse, _ = self.model(
            #     self.input_data, self.g_spatial, self.g_expr, 
            #     stage='finetune', freeze_encoder=self.freeze_encoder
            # )
            h1_proj, h2_proj, h_common_proj, z_final, x_rec, Gf_sparse, gate_weights= self.model(
                self.input_data, self.g_spatial, self.g_expr, 
                stage='finetune', freeze_encoder=self.freeze_encoder
            )
            
            # === 计算损失 ===
            # 1. 重构损失
            loss_rec = self.rec_crit(x_rec, self.input_data)

            loss_smooth = torch.mean(torch.norm(z_final - torch.spmm(self.g_spatial, z_final),p=2,dim=1))
            
            # 2. 结构引导对比损失
            loss_contrast1 = self.contrast_crit(h1_proj, h_common_proj, Gf_sparse)
            loss_contrast2 = self.contrast_crit(h2_proj, h_common_proj, Gf_sparse)
            loss_contrast = (loss_contrast1 + loss_contrast2) / 2
            
            # 3. 聚类损失 (保留代码但权重=0)
            if self.beta > 0:
                label1 = self.model.projectClsHead(h1_proj)
                label2 = self.model.projectClsHead(h2_proj)
                loss_cluster = self.cluster_crit(label1, label2)
            else:
                loss_cluster = 0.0
            
            # 总损失
            loss = self.gamma * loss_rec + self.kappa * loss_contrast + self.beta * loss_cluster + self.theta*loss_smooth
            
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Train Epoch {epoch+1}: Loss = {loss.item():.6f},Rec Loss = {loss_rec.item():.6f}, Contrast Loss = {loss_contrast.item():.6f},  Cluster Loss = {loss_cluster if isinstance(loss_cluster, float) else loss_cluster.item():.6f}, Smooth Loss = {loss_smooth.item():.6f}")
            
            # === 定期评估 (每100轮) ===
            if (epoch + 1) % 100 == 0:
                print(f"\n  Finetune Epoch {epoch+1}:")
                print(f"    Total Loss   = {loss.item():.4f}")
                print(f"    Rec Loss     = {loss_rec.item():.4f}")
                print(f"    Contrast Loss = {loss_contrast.item():.4f}")
                print(f"    Smooth Loss = {loss_smooth.item():.4f}")
                
                # 如果有聚类损失,检查视图一致性
                if self.beta > 0:
                    predLabel1_np = label1.detach().cpu().numpy().argmax(axis=1)
                    predLabel2_np = label2.detach().cpu().numpy().argmax(axis=1)
                    cur_ari = adjusted_rand_score(predLabel1_np, predLabel2_np)
                    print(f"    View ARI     = {cur_ari:.4f}")
                    
                    # === 早停机制 (关键!) ===
                    if cur_ari >= max_ari:
                        print(f"\n✓ Early stopping triggered! (ARI={cur_ari:.4f} >= {max_ari:.2f})")
                        predLabel, features_fuse, gate_weights = self.updateResult()
                        return predLabel, features_fuse, gate_weights
                
                # 打印门控统计
                self.model.eval()
                with torch.no_grad():
                    self.model(self.input_data, self.g_spatial, self.g_expr, 
                              stage='finetune', freeze_encoder=True)
                    stats = self.model.gate_stats
                    print(f"    Gate: spatial={stats['spatial_mean']:.3f}±{stats['spatial_std']:.3f}, "
                          f"expr={stats['expr_mean']:.3f}±{stats['expr_std']:.3f}")
                self.model.train()
        
        print("\n✓ Finetune finished (max epochs reached)")
        
        # 如果没有早停,返回最终结果
        predLabel, features_fuse, gate_weights = self.updateResult()
        return predLabel, features_fuse, gate_weights
    
    def train(self):
        """
        完整的两阶段训练流程
        
        Returns:
            predLabel: [N] 聚类标签
            features_fuse: [N, D] 融合嵌入
        """
        # 阶段1: 预训练
        self.pretrain()
        
        # 阶段2: 训练 (可能触发早停)
        predLabel, features_fuse, gated_weights = self.finetune()
        
        return predLabel, features_fuse, gated_weights