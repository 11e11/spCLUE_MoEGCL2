import torch
from torch import  nn
import math
import torch.nn.functional as F


# class StructureGuidedContrastiveLoss(nn.Module):
#     """
#     结构引导的对比学习 (基于MoEGCL)
#     正样本: 融合图中的邻居
#     负样本: 融合图中的非邻居
#     """
    
#     def __init__(self, temperature=0.2, device='cuda'):
#         super().__init__()
#         self.temperature = temperature
#         self.device = device
#         self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
#     def forward(self, z_view, z_fused, Gf_sparse, eps=1e-8):
#         """
#         Args:
#             z_view: [N, D] 视图特定编码 (Z1 or Z2)
#             z_fused: [N, D] 融合编码
#             Gf_sparse: 融合图 (用于定义正负样本)
        
#         Returns:
#             loss: scalar
#         """
#         N = z_view.size(0)
        
#         # 归一化
#         # z_view = F.normalize(z_view, p=2, dim=1)
#         # z_fused = F.normalize(z_fused, p=2, dim=1)
        
#         # 计算相似度矩阵
#         sim_matrix = torch.matmul(z_view, z_fused.T) / self.temperature  # [N, N]
        
#         # 从融合图中提取邻居关系
#         Gf_dense = Gf_sparse.to_dense()  # [N, N]
        
#         # 正样本: 同一节点 (对角线)
#         positive_mask = torch.eye(N).to(self.device)
        
#         # 增强正样本: 融合图中的邻居也视为正样本
#         # 注意: 这里可以调整,只用对角线也可以
#         positive_mask = positive_mask + Gf_dense
#         positive_mask = (positive_mask > 0).float()
#         positive_mask.fill_diagonal_(1)  # 确保对角线为1
        
#         # 负样本: 所有非正样本
#         negative_mask = 1 - positive_mask
        
#         # 提取正样本得分
#         positive_scores = (sim_matrix * positive_mask).sum(dim=1, keepdim=True) / \
#                          (positive_mask.sum(dim=1, keepdim=True) + eps)
        
#         # 提取负样本得分
#         negative_scores = sim_matrix * negative_mask
        
#         # 构建logits (正样本在第一列)
#         logits = torch.cat([positive_scores, negative_scores], dim=1)
        
#         # 标签: 正样本是第0类
#         labels = torch.zeros(N).to(self.device).long()
        
#         # InfoNCE损失
#         loss = self.criterion(logits, labels) / N
        
#         return loss
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


import torch
import torch.nn as nn

class StructureGuidedContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0, device='cuda', chunk_size=1024): # 4090建议调大到1024
        super().__init__()
        self.temperature = temperature
        self.chunk_size = chunk_size

    def forward(self, h_view, h_common, Gf_sparse):
        device = h_view.device
        N = h_view.size(0)

        # 特征归一化（对比学习标准操作，防止点积数值过大）
        h = torch.cat([h_view, h_common], dim=0) 
        h = torch.nn.functional.normalize(h, dim=1)

        # 提取图邻居（预处理索引，加速查找）
        if Gf_sparse.is_sparse:
            Gf_sparse = Gf_sparse.coalesce()
            row, col = Gf_sparse.indices()
        else:
            row, col = torch.nonzero(Gf_sparse > 0, as_tuple=True)

        loss_sum = 0.0
        
        for start in range(0, 2 * N, self.chunk_size):
            end = min(start + self.chunk_size, 2 * N)
            rows_global = torch.arange(start, end, device=device)
            h_chunk = h[start:end]

            # 1. 计算相似度 (chunk, 2N)
            sim = torch.matmul(h_chunk, h.T) / self.temperature
            
            # 2. 准备掩码
            mask = torch.zeros_like(sim, dtype=torch.bool)
            
            # self mask & positive mask
            pos_idx = rows_global.clone()
            pos_idx[rows_global < N] += N
            pos_idx[rows_global >= N] -= N
            
            mask[torch.arange(end - start), rows_global] = True
            mask[torch.arange(end - start), pos_idx] = True

            # 3. 结构邻居 mask (优化版：避免线性扫描)
            node_ids = rows_global % N
            # 找出 row 中所有落在当前 chunk 范围内的 edges
            # 这一步能大幅加速大规模数据的训练
            relevant_edges = torch.isin(row, node_ids)
            if relevant_edges.any():
                r_sub = row[relevant_edges]
                c_sub = col[relevant_edges]
                
                # 建立 node_id 到 chunk 内相对索引的映射
                # 这种方式比 for 循环中执行 row == node 快得多
                rel_map = torch.zeros(N, dtype=torch.long, device=device)
                rel_map[node_ids] = torch.arange(end - start, device=device)
                
                rows_in_mask = rel_map[r_sub]
                mask[rows_in_mask, c_sub] = True      # 遮蔽 View 1 邻居
                mask[rows_in_mask, c_sub + N] = True  # 遮蔽 Common 邻居

            # 4. 正样本相似度（单独提取）
            pos_sim = torch.sum(h_chunk * h[pos_idx], dim=1) / self.temperature

            # 5. 计算带掩码的 LogSumExp
            sim = sim.masked_fill(mask, float('-inf'))
            logits_max = torch.max(sim, dim=1, keepdim=True)[0]
            # 修正：如果一行全是邻居导致 max 是 -inf，设为 0
            logits_max = torch.where(torch.isinf(logits_max), torch.zeros_like(logits_max), logits_max)

            exp_sim_neg = torch.exp(sim - logits_max)
            sum_exp_neg = exp_sim_neg.sum(dim=1)
            
            # 修正公式：正样本必须放回分母中以保持与原代码一致
            exp_pos = torch.exp(pos_sim - logits_max.squeeze())
            
            # Loss = -log( exp(pos) / (exp(pos) + sum(exp(neg))) )
            current_loss = -(pos_sim - logits_max.squeeze() - torch.log(exp_pos + sum_exp_neg + 1e-8))

            loss_sum += current_loss.sum()

        return loss_sum / (2 * N)
# class StructureGuidedContrastiveLoss(nn.Module):
#     def __init__(self, temperature=1.0, device='cuda'):
#         super().__init__()
#         self.temperature = temperature
#         self.device = device

#     def forward(self, h_view, h_common, Gf_sparse):
#         """
#         显存友好且支持稀疏/稠密双后端的结构引导对比损失
#         """
#         N = h_view.size(0)
        
#         # 1. 计算相似度矩阵 (2N x 2N)
#         h = torch.cat((h_view, h_common), dim=0) 
#         sim = torch.matmul(h, h.T) / self.temperature 
        
#         # 2. 构造掩码矩阵
#         mask_inf = torch.zeros(2 * N, 2 * N, device=self.device)
        
#         # --- 核心：提取结构邻居掩码 ---
#         if Gf_sparse.is_sparse:
#             Gf_sparse = Gf_sparse.coalesce()
#             indices = Gf_sparse._indices()
#             row, col = indices[0], indices[1]
#             # 排除 4 个象限中的结构邻居
#             mask_inf[:N, :N][row, col] = -1e9
#             mask_inf[N:, N:][row, col] = -1e9
#             mask_inf[:N, N:][row, col] = -1e9
#             mask_inf[N:, :N][row, col] = -1e9
#         else:
#             neighbor_mask = Gf_sparse > 0
#             mask_inf[:N, :N][neighbor_mask] = -1e9
#             mask_inf[N:, N:][neighbor_mask] = -1e9
#             mask_inf[:N, N:][neighbor_mask] = -1e9
#             mask_inf[N:, :N][neighbor_mask] = -1e9
            
#         # --- 统一处理：对角线和正样本对 ---
#         diag_indices = torch.arange(N, device=self.device)
#         # 排除同视图自相似 (对角线)
#         mask_inf[:N, :N][diag_indices, diag_indices] = -1e9
#         mask_inf[N:, N:][diag_indices, diag_indices] = -1e9
#         # 排除跨视图正样本对 (用于分母)
#         mask_inf[:N, N:][diag_indices, diag_indices] = -1e9
#         mask_inf[N:, :N][diag_indices, diag_indices] = -1e9
        
#         # 3. 计算 InfoNCE (共享逻辑)
#         sim_with_mask = sim + mask_inf
        
#         # 提取正样本 (跨视图对角线)
#         pos_i_j = torch.diag(sim[:N, N:]) 
#         pos_j_i = torch.diag(sim[N:, :N])
#         positives = torch.cat([pos_i_j, pos_j_i], dim=0).view(2 * N, 1)
        
#         # 数值稳定的 LogSumExp
#         logits_max, _ = torch.max(sim_with_mask, dim=1, keepdim=True)
#         exp_sim = torch.exp(sim_with_mask - logits_max)
#         sum_exp_neg = exp_sim.sum(dim=1, keepdim=True)
#         exp_pos = torch.exp(positives - logits_max)
        
#         loss = - (positives - logits_max) + torch.log(exp_pos + sum_exp_neg)
        
#         return loss.mean()
# class StructureGuidedContrastiveLoss(nn.Module):
#     def __init__(self, temperature=1.0, device='cuda'):
#         super().__init__()
#         self.temperature = temperature
#         self.device = device

#     def forward(self, h_view, h_common, Gf_dense):
#         """
#         优化后的显存友好版
#         """
#         N = h_view.size(0)
        
#         # 1. 拼接特征并计算相似度矩阵 (2N x 2N)
#         # 这是避不开的 O(N^2)，但我们要尽量减少它的副本
#         h = torch.cat((h_view, h_common), dim=0)  # [2N, D]
#         sim = torch.matmul(h, h.T) / self.temperature  # [2N, 2N]
        
#         # 2. 构造结构引导的惩罚 (Structural Penalty)
#         # 不要使用 repeat(2,2) 或 ones - S_1，那会创建巨大的新矩阵
#         # 直接利用 Gf_dense [N, N] 的四块拼接逻辑进行原地操作或掩码
        
#         # 我们要排除：
#         # (a) 对角线 (self-similarity)
#         # (b) 正样本对 (h_view[i] vs h_common[i])
#         # (c) 图结构中的邻居 (Structural Neighbors)
        
#         # 构造一个极小值掩码，将不需要作为负样本的位置设为 -1e9
#         # 这样在计算 Softmax (CrossEntropy) 时这些位置的贡献几乎为 0
#         mask_inf = torch.zeros(2 * N, 2 * N, device=self.device)
        
#         # 处理 (c) 邻居节点：利用广播直接填充 4 个象限
#         # 假设 Gf_dense 中 1 表示邻居
#         neighbor_mask = Gf_dense > 0
#         mask_inf[:N, :N][neighbor_mask] = -1e9
#         mask_inf[N:, N:][neighbor_mask] = -1e9
#         mask_inf[:N, N:][neighbor_mask] = -1e9
#         mask_inf[N:, :N][neighbor_mask] = -1e9
        
#         # 处理 (a) & (b) 对角线和正样本对
#         diag_mask = torch.eye(N, device=self.device).bool()
#         # 排除同视图对角线
#         mask_inf[:N, :N][diag_mask] = -1e9
#         mask_inf[N:, N:][diag_mask] = -1e9
#         # 排除跨视图正样本（这些我们要单独提取，不作为负样本）
#         mask_inf[:N, N:][diag_mask] = -1e9
#         mask_inf[N:, :N][diag_mask] = -1e9
        
#         # 将掩码应用到相似度矩阵
#         sim_with_mask = sim + mask_inf
        
#         # 3. 提取正样本 (2N, 1)
#         # sim[:N, N:] 的对角线是 h_view[i] vs h_common[i]
#         pos_i_j = torch.diag(sim[:N, N:]) 
#         pos_j_i = torch.diag(sim[N:, :N])
#         positives = torch.cat([pos_i_j, pos_j_i], dim=0).view(2 * N, 1)
        
#         # 4. 提取负样本 (2N, 2N-2-neighbors)
#         # 关键优化：不再使用 sim[mask].reshape，因为那样会产生不规则形状，显存开销极大
#         # 我们改用 LogSumExp 的逻辑来手动计算 InfoNCE
        
#         # 计算分母：sum(exp(pos) + sum(exp(neg)))
#         # 为了数值稳定，使用如下公式：
#         # Loss = -pos + log(sum(exp(all_sim_in_row)))
        
#         logits_max, _ = torch.max(sim_with_mask, dim=1, keepdim=True)
#         # 减去最大值防止溢出
#         exp_sim = torch.exp(sim_with_mask - logits_max)
#         sum_exp_neg = exp_sim.sum(dim=1, keepdim=True)
        
#         # 加上正样本的 exp
#         exp_pos = torch.exp(positives - logits_max)
        
#         # InfoNCE = -log( exp(pos) / (exp(pos) + sum(exp(neg))) )
#         loss = - (positives - logits_max) + torch.log(exp_pos + sum_exp_neg)
        
#         return loss.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, x, xbar, eps=1e-8):
        posScores = torch.exp((x * xbar).sum(dim=1) / self.temperature)
        negScores = torch.exp((x @ xbar.T) / self.temperature).sum(dim=1)
        return -torch.log(posScores / (negScores + eps)).mean()


class CCRLoss(nn.Module):
    """
    DICR Loss (Dual Information Correlation Reduction)
    特征级去相关损失 - 对齐MAFN实现
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, xbar):
        """
        x: 视图1的嵌入 (N, d), 例如空间图嵌入 E_s
        xbar: 视图2的嵌入 (N, d), 例如特征图嵌入 E_f
        
        Returns:
            特征级去相关损失
        """
        # 🔥 关键修改：转置到特征空间
        # x: [N, d] -> x.t(): [d, N]
        S = self._cross_correlation(x.t(), xbar.t())  # [d, d]
        
        # 对角线损失：同一特征维度在不同视图应对齐 (S[i,i]→1)
        diag_loss = torch.diagonal(S).add(-1).pow(2).mean()
        
        # 非对角线损失：不同特征维度应不相关 (S[i,j]→0)
        off_diag_loss = self._off_diagonal(S).pow(2).mean()
        
        return diag_loss + off_diag_loss
    
    def _cross_correlation(self, Z_v1, Z_v2):
        """
        计算跨视图特征相关矩阵
        Args:
            Z_v1: [d, N] - 视图1的特征 (转置后)
            Z_v2: [d, N] - 视图2的特征 (转置后)
        Returns:
            S: [d, d] - 特征相关矩阵
        """
        # L2归一化后计算内积
        Z_v1_norm = F.normalize(Z_v1, p=2, dim=1)  # [d, N]
        Z_v2_norm = F.normalize(Z_v2, p=2, dim=1)  # [d, N]
        return torch.mm(Z_v1_norm, Z_v2_norm.t())  # [d, d]
    
    def _off_diagonal(self, x):
        """
        提取矩阵的非对角线元素
        Args:
            x: [d, d] 方阵
        Returns:
            非对角线元素的扁��化向量
        """
        n, m = x.shape
        assert n == m, "Matrix must be square"
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class ClusterLoss(nn.Module):
    def __init__(
            self,
            n_classes,
            device=torch.device("cuda:0"),
            temperature=0.2,
    ):
        super(ClusterLoss, self).__init__()
        self.n_classes = n_classes
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(n_classes)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, n_classes):
        N = 2 * n_classes
        mask = torch.ones(N, N)
        mask.fill_diagonal_(0)
        for i in range(n_classes):
            mask[i, i + n_classes] = 0
            mask[i + n_classes, i] = 0
        mask = mask.bool()
        return mask

    def normalizeLabel(self, c_i, c_j):
        c_i = torch.square(c_i)
        c_j = torch.square(c_j)
        p_i = c_i.sum(dim=0).view(-1)
        c_i /= p_i
        p_i = c_i.sum(dim=1).view(-1)
        c_i /= p_i.unsqueeze(1)
        p_j = c_j.sum(dim=0).view(-1)
        c_j /= p_j
        p_j = c_j.sum(dim=1).view(-1)
        c_j /= p_j.unsqueeze(1)
        return c_i, c_j

    def forward(self, c_i, c_j):
        p_i = c_i.sum(dim=0).view(-1)
        p_i /= p_i.sum()
        neg_entropy_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        neg_entropy_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        neg_entropy_loss = neg_entropy_i + neg_entropy_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.n_classes
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1),
                                c.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.n_classes)
        sim_j_i = torch.diag(sim, -self.n_classes)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=-1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + 1. * neg_entropy_loss

class GraphConsis(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, emb, graphWeight):
        dist1 = torch.cdist(emb, emb, p=2)
        dist1 = torch.div(dist1, torch.max(dist1))
        return torch.mean((1 - dist1) * graphWeight)


class GraphRecLoss(nn.Module):
    def __init__(self, norm_val, pos_weight) -> None:
        super().__init__()
        self.norm_val = norm_val
        self.pos_weight = pos_weight

    def forward(self, emb, target):
        # emb = F.normalize(emb, p=2, dim=1)
        input = emb @ emb.T
        logits = F.binary_cross_entropy_with_logits(input,
                                                    target,
                                                    pos_weight=self.pos_weight)
        return self.norm_val * logits


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, xbar):
        return torch.square(x - xbar).mean(dim=1).mean()


class ZINBLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _nan2inf(self, x):
        """处理计算中可能出现的 NaN，将其转为较大的常数"""
        return torch.where(torch.isnan(x), torch.full_like(x, 1e6), x)

    def forward(self, x, mean, disp, pi, ridge_lambda=0.0):
        """
        x:    对应 adata.X (Normalized & Scaled)
        mean: 解码器输出的均值
        disp: 解码器输出的离散度
        pi:   解码器输出的零概率
        """
        eps = 1e-10
        
        # 1. NB 对数似然计算 (处理 y_true + 1.0 用于 Gamma 函数)
        # 虽然 x 是连续值，但 torch.lgamma 支持实数输入
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        
        # 稳定性优化：利用 log(1 + x/y) 避免直接除法
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (
              x * (torch.log(disp + eps) - torch.log(mean + eps)))
        
        nb_final = t1 + t2
        
        # 2. 考虑零膨胀的情况
        # NB case: 概率为 (1-pi) * P_nb(x)
        nb_case = nb_final - torch.log(1.0 - pi + eps)
        
        # Zero case: 概率为 pi + (1-pi) * P_nb(0)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        
        # 根据 x 是否接近 0 选择损失
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        # 3. 正则项
        if ridge_lambda > 0:
            result += ridge_lambda * torch.square(pi)

        # 4. 最终均值化与 NaN 保护
        result = torch.mean(result)
        result = self._nan2inf(result)
        
        return result
class GraphReconstructionLoss(nn.Module):
    """改进版图重构损失 - 直接处理重构矩阵"""
    
    def __init__(self, sample_strategy='hybrid', neg_sample_ratio=1.0):
        super().__init__()
        self.sample_strategy = sample_strategy
        self.neg_sample_ratio = neg_sample_ratio
    
    def forward(self, adj_rec, adj_true, emb=None, pos_weight=None, norm_val=1.0):
        """
        Args:
            adj_rec: [N, N] 模型输出的重构矩阵（已sigmoid）或 None
            adj_true: sparse tensor 真实邻接矩阵
            emb: [N, D] 节点嵌入（仅edge策略需要）
            pos_weight: 正样本权重
            norm_val: 归一化系数
        """
        N = adj_true.size(0)
        
        # 自动计算权重
        if pos_weight is None:
            num_pos = adj_true._values().sum().item()
            num_neg = N * N - num_pos
            pos_weight = torch.tensor(num_neg / (num_pos + 1e-8), 
                                     device=adj_true.device)
        
        # ===== 策略1: 完整图重构（使用模型输出的矩阵）=====
        if self.sample_strategy == 'full' or N < 5000:
            if adj_rec is None:
                raise ValueError("adj_rec不能为None（full/hybrid策略）")
            
            adj_dense = adj_true.to_dense()
            
            # 直接使用模型输出的adj_rec（已经sigmoid过）
            loss = F.binary_cross_entropy(
                adj_rec, adj_dense, 
                weight=self._get_weight_matrix(adj_dense, pos_weight)
            )
            return norm_val * loss
        
        # ===== 策略2: 边采样（需要嵌入重新计算）=====
        elif self.sample_strategy == 'edge':
            if emb is None:
                raise ValueError("edge策略需要传入emb参数")
            
            pos_edges = adj_true._indices()
            pos_scores = (emb[pos_edges[0]] * emb[pos_edges[1]]).sum(dim=1)
            pos_labels = torch.ones_like(pos_scores)
            
            num_neg = int(pos_edges.size(1) * self.neg_sample_ratio)
            neg_edges = self._negative_sampling(N, num_neg, pos_edges, emb.device)
            neg_scores = (emb[neg_edges[0]] * emb[neg_edges[1]]).sum(dim=1)
            neg_labels = torch.zeros_like(neg_scores)
            
            all_scores = torch.cat([pos_scores, neg_scores])
            all_labels = torch.cat([pos_labels, neg_labels])
            
            # 注意：这里用logits（未sigmoid），所以用with_logits版本
            loss = F.binary_cross_entropy_with_logits(
                all_scores, all_labels, 
                pos_weight=pos_weight
            )
            return norm_val * loss
        
        # ===== 策略3: 混合采样 =====
        else:  # 'hybrid'
            if adj_rec is None:
                raise ValueError("adj_rec不能为None（full/hybrid策略）")
            
            sample_size = min(2000, N)
            idx = torch.randperm(N, device=adj_rec.device)[:sample_size]
            
            # 从完整重构矩阵中采样子矩阵
            sub_adj_rec = adj_rec[idx][:, idx]  # [sample_size, sample_size]
            sub_adj_true = self._extract_subgraph(adj_true, idx)
            
            loss = F.binary_cross_entropy(
                sub_adj_rec, sub_adj_true,
                weight=self._get_weight_matrix(sub_adj_true, pos_weight)
            )
            return norm_val * loss
    
    def _get_weight_matrix(self, adj, pos_weight):
        """生成权重矩阵"""
        weight = torch.ones_like(adj)
        weight[adj == 1] = pos_weight
        return weight
    
    def _negative_sampling(self, N, num_neg, pos_edges, device):
        """负采样（仅edge策略使用）"""
        pos_set = set(map(tuple, pos_edges.T.cpu().numpy()))
        neg_edges = []
        
        max_attempts = num_neg * 10  # 防止死循环
        attempts = 0
        
        while len(neg_edges) < num_neg and attempts < max_attempts:
            src = torch.randint(0, N, (1,), device=device).item()
            dst = torch.randint(0, N, (1,), device=device).item()
            
            if (src, dst) not in pos_set and src != dst:
                neg_edges.append([src, dst])
            attempts += 1
        
        if len(neg_edges) < num_neg:
            print(f"警告: 只采样到{len(neg_edges)}/{num_neg}个负样本")
        
        return torch.tensor(neg_edges, device=device).T
    
    def _extract_subgraph(self, adj_sparse, idx):
        """提取子图的稠密邻接矩阵"""
        N = len(idx)
        indices = adj_sparse._indices()
        values = adj_sparse._values()
        
        # 创建索引映射
        idx_cpu = idx.cpu().numpy()
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(idx_cpu)}
        
        # 筛选属于子图的边
        mask = []
        new_indices = []
        
        for i in range(indices.size(1)):
            src = indices[0, i].item()
            dst = indices[1, i].item()
            
            if src in idx_map and dst in idx_map:
                mask.append(i)
                new_indices.append([idx_map[src], idx_map[dst]])
        
        if len(new_indices) == 0:
            # 子图无边（极端情况）
            return torch.zeros((N, N), device=adj_sparse.device)
        
        new_indices = torch.tensor(new_indices, device=adj_sparse.device).T
        new_values = values[mask]
        
        sub_adj_sparse = torch.sparse_coo_tensor(new_indices, new_values, (N, N))
        return sub_adj_sparse.to_dense()