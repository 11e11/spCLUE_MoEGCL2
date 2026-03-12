# import scanpy as sc
# import numpy as np
# import scipy.sparse as sp
# from scipy.spatial.distance import cdist
# from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph, kneighbors_graph
# from sklearn.preprocessing import normalize
# from sklearn.utils import issparse

# def preprocess(adata, hvgNumber=3000):
#     """
#     完全对齐 MAFN 的预处理逻辑:
#     1. 过滤基因 (min_cells=100)
#     2. 筛选 HVG (seurat_v3)
#     3. 归一化 (X / sum * 10000)
#     4. Log1p
#     5. Scale (zero_center=False, max_value=10)
#     """
#     print(f"MAFN-style preprocessing (HVG={hvgNumber}) ---------------->")
    
#     # 1. 过滤基因: MAFN 使用 min_cells=100
#     sc.pp.filter_genes(adata, min_cells=100)
    
#     # 备份层
#     if issparse(adata.X):
#         adata.layers['count'] = adata.X.copy()
#     else:
#         adata.layers['count'] = adata.X.copy()

#     # 2. 筛选高变基因
#     print(f"========== selecting {hvgNumber} HVGs ============")
#     sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="count", n_top_genes=hvgNumber, subset=True)
    
#     # 3. 归一化: MAFN 逻辑是 X / sum * 10000 (等价于 normalize_total)
#     sc.pp.normalize_total(adata, target_sum=1e4)
    
#     # 4. Log1p 处理
#     sc.pp.log1p(adata)
    
#     # 5. Scale 处理: 不居中且截断为 10
#     sc.pp.scale(adata, zero_center=False, max_value=10)
    
#     return adata

# def symm_norm(adj, weightDiag=1.0, eps=1e-8):
#     """
#     完全对齐 MAFN/GCN 的标准归一化逻辑:
#     A_hat = A + I (即 weightDiag=1.0)
#     Return: D^-1/2 * A_hat * D^-1/2
#     """
#     if not sp.issparse(adj):
#         adj = sp.coo_matrix(adj)
    
#     n_spot = adj.shape[0]
#     adj = adj.tocoo()
    
#     # MAFN 逻辑中，邻居边的权重不缩放，保持 1.0
#     adj_rows = adj.row
#     adj_cols = adj.col
#     adj_data = adj.data # 默认为 1.0
    
#     # 添加权重为 1.0 的自环 (weightDiag = 1.0)
#     diag_rows = np.arange(n_spot)
#     diag_cols = np.arange(n_spot)
#     diag_data = np.full(n_spot, weightDiag) 
    
#     final_rows = np.concatenate([adj_rows, diag_rows])
#     final_cols = np.concatenate([adj_cols, diag_cols])
#     final_data = np.concatenate([adj_data, diag_data])
    
#     # 构建 A + I
#     adj_self = sp.coo_matrix((final_data, (final_rows, final_cols)), shape=(n_spot, n_spot))

#     # 计算度矩阵 D_hat
#     degrees = np.array(adj_self.sum(axis=1)).flatten()
#     degrees = 1. / np.sqrt(degrees + eps)
    
#     # 矩阵计算: D^-1/2 * (A+I) * D^-1/2
#     adj_self = adj_self.tocsr()
#     adj_self = adj_self.multiply(degrees[:, None]) 
#     adj_self = adj_self.multiply(degrees[None, :]) 
    
#     return adj_self

# def prepare_graph(adata, key="spatial", n_neighbors=15, radius=550, metric="cosine", self_weight=1.0):
#     """
#     完全对齐 MAFN 的构图逻辑:
#     - 空间图: 使用物理坐标的 Radius Graph
#     - 特征图: 直接使用预处理后的 adata.X (HVG) 进行 KNN, 不用 PCA
#     """
#     if key == "spatial":
#         print(f"正在构建空间图: 使用物理半径 radius={radius} ...")
#         X_pos = adata.obsm[key]
#         # MAFN 风格的空间构图
#         adj = radius_neighbors_graph(X_pos, radius=radius, mode='connectivity', metric='euclidean', include_self=False)
#     else: 
#         print(f"正在构建表达图: 使用 HVG 矩阵, KNN k={n_neighbors}, 度量={metric} ...")
#         # 核心修改: 直接使用 adata.X, 即使是稀疏矩阵, sklearn 的 kneighbors_graph 也支持
#         # MAFN 默认 k=15, metric='cosine'
#         X_feat = adata.X
        
#         adj = kneighbors_graph(X_feat, n_neighbors=n_neighbors, mode='connectivity', metric=metric, include_self=False)

#     # 对称化处理 (MAFN 逻辑: A = A + A.T...)
#     print("  -> 对称化与归一化...")
#     adj = adj.tocoo()
#     # 只要 (i,j) 或 (j,i) 有边，则对称化后均有边，且二值化为 1
#     sym_graph = adj + adj.T 
#     sym_graph.data = np.ones_like(sym_graph.data)
    
#     # 归一化
#     norm_adj = symm_norm(sym_graph, weightDiag=self_weight)
    
#     print(f"{key} graph created successfully <----\n")
#     return norm_adj
import scanpy as sc
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import numpy as np
import scipy.sparse as sp
from sklearn.utils import issparse



def preprocess(adata, hvgNumber=None):
    print("normalized data ---------------->")
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    if issparse(adata.X):
        adata.layers['count'] = adata.X.copy() # 保持稀疏以节省内存
    else:
        adata.layers['count'] = adata.X.copy()
    if not hvgNumber is None:
        print(f"========== selecting HVG ============")
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="count",n_top_genes=hvgNumber, subset=False)
        adata = adata[:, adata.var["highly_variable"] == True]
        sc.pp.scale(adata)
        return adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    return adata



import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

def symm_norm(adj, weightDiag=.3, eps=1e-8):
    '''
    保持原有的归一化逻辑不变
    return: D^{-1/2} (A + I) D^{-1 / 2}
    '''
    # 确保是稀疏矩阵操作以节省内存
    if not sp.issparse(adj):
        adj = sp.coo_matrix(adj)
    
    n_spot = adj.shape[0]
    # 对角线处理：(1-w)*A + w*I
    # 注意：这里为了效率，我们不直接生成 dense 矩阵，而是利用稀疏矩阵特性
    adj = adj.tocoo()
    
    # 重新构建带权重的邻接矩阵（包含对角线）
    # 逻辑：非对角线元素 * (1 - weightDiag)
    adj_data = adj.data * (1 - weightDiag)
    adj_rows = adj.row
    adj_cols = adj.col
    
    # 添加对角线元素
    diag_rows = np.arange(n_spot)
    diag_cols = np.arange(n_spot)
    diag_data = np.full(n_spot, weightDiag)
    
    # 合并
    final_rows = np.concatenate([adj_rows, diag_rows])
    final_cols = np.concatenate([adj_cols, diag_cols])
    final_data = np.concatenate([adj_data, diag_data])
    
    adj_self = sp.coo_matrix((final_data, (final_rows, final_cols)), shape=(n_spot, n_spot))

    # 计算度矩阵
    # sum(axis=1) 对于稀疏矩阵返回的是 np.matrix，需要转为 array
    degrees = np.array(adj_self.sum(axis=1)).flatten()
    degrees = 1. / np.sqrt(degrees + eps)
    
    # D^{-1/2} A D^{-1/2}
    # 利用对角矩阵乘法的性质：行缩放和列缩放
    # CSR 格式做乘法更高效
    adj_self = adj_self.tocsr()
    
    # 每一行乘以 degrees (左乘对角矩阵)
    sp.diags(degrees) @ adj_self @ sp.diags(degrees)
    
    # 注意：scipy 稀疏矩阵乘法会自动处理 broadcasting
    # 更高效的手动实现：
    adj_self = adj_self.multiply(degrees[:, None]) # 乘行因子
    adj_self = adj_self.multiply(degrees[None, :]) # 乘列因子
    
    return adj_self


def prepare_graph(adata, key="spatial", n_neighbors=8, n_comps=50, 
                  metric="cosine",  # 新增参数：cosine 或 euclidean
                  svd_solver="randomized", self_weight=0.3):
    
    n_spots = adata.shape[0]
    print(f"正在构建图: {key}, 使用度量: {metric} ...")

    # ==========================================
    # 1. 准备特征数据 (Feature Preparation)
    # ==========================================
    if key == "spatial":
        # 空间图通常依然使用欧氏距离（物理距离）
        X_data = adata.obsm[key]
        use_metric = 'euclidean' 
        print("  -> 使用空间坐标 (euclidean)")
        
    else: # expr
        print("  -> 使用 PCA 表达特征")
        # 计算 PCA
        if 'X_pca' in adata.obsm:
             X_data = adata.obsm['X_pca'][:, :n_comps]
        else:
            X_data = PCA(n_components=n_comps, random_state=0, svd_solver=svd_solver).fit_transform(adata.X)
        
        # 处理度量标准
        if metric == "cosine":
            # 技巧：余弦距离等价于 L2 归一化后的欧氏距离
            # 先对向量做 L2 归一化，后续直接用 KNN 的 euclidean 搜索，速度极快
            X_data = normalize(X_data, norm='l2', axis=1)
            use_metric = 'euclidean' 
        else:
            use_metric = 'euclidean' # 默认为欧氏距离

    # ==========================================
    # 2. 构建 KNN 图 (Efficient KNN Construction)
    # ==========================================
    print("  -> 计算最近邻 (NearestNeighbors)...")
    
    # 使用 sklearn 的算法，复杂度 O(N log N)，避免 O(N^2) 矩阵
    # n_neighbors + 1 是因为 KNN 会把自己算作第1个邻居（距离为0），需要剔除
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=use_metric, algorithm='auto')
    nbrs.fit(X_data)
    
    # mode='connectivity' 直接返回稀疏矩阵 (CSR)，且只有 0/1 (二值化)
    # 这对应了原代码中的 adjBip = np.where(weights > adjFilter, 1, 0)
    # 且不再需要手动 sort 和 threshold，sklearn 全帮我们做好了
    knn_graph = nbrs.kneighbors_graph(X_data, mode='connectivity')
    
    # 剔除对角线 (自己连自己)
    knn_graph.setdiag(0)
    knn_graph.eliminate_zeros()
    
    # ==========================================
    # 3. 对称化处理 (Symmetrization)
    # ==========================================
    # 逻辑：A 是 B 的邻居，但 B 不一定是 A 的。我们取并集或交集。
    # 原代码逻辑：(W + W.T) / 2 -> 只要有一边连了，权重就变 0.5。
    # 这里我们做逻辑 OR：只要是单向邻居，就视为相连 (权重为1)
    
    print("  -> 对称化与归一化...")
    # 转为 COO 以便相加
    knn_graph = knn_graph.tocoo()
    # 这种加法会让双向邻居变成 2，单向邻居变成 1
    sym_graph = knn_graph + knn_graph.T 
    
    # 重新二值化：只要大于0就是邻居
    sym_graph.data = np.ones_like(sym_graph.data)
    
    # ==========================================
    # 4. 归一化 (Normalization)
    # ==========================================
    # 调用改进后的稀疏归一化函数
    norm_adj = symm_norm(sym_graph, weightDiag=self_weight)
    
    print(f"{key} graph created successfully <----\n")
    return norm_adj
