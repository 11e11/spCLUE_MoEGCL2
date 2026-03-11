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


def calcGAEParams(graph, n_samples):
    '''graph is a bipartite graph, return pos_weight and norm_val
    '''
    non_zero_cnt = graph.sum()
    norm_val = (n_samples * n_samples) / (2 * (n_samples * n_samples - non_zero_cnt))
    pos_weight = (n_samples * n_samples - non_zero_cnt) / non_zero_cnt
    return norm_val, pos_weight


def calcGraphWeight(coor, eps=1e-6):
    dist = cdist(coor, coor, "euclidean")
    dist = dist / (np.max(dist) + eps)
    return dist


def correlation_graph(A, B):
    '''calculate correlation between A and B.
    Args:
        A (np.ndarray): sample matrix, shape: [samples, features].
        B (np.ndarray): sample matrix, shape: [samples, features].
    Returns: 
        corr (np.ndarray): correlation matrix of features, shape: [features, features].
    '''
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)
    return am.T @ bm / (np.sqrt(np.sum(am**2, axis=0, keepdims=True)).T * np.sqrt(np.sum(bm**2, axis=0, keepdims=True)))


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


def prepare_graph(adata, key="spatial", n_neighbors=12, n_comps=50, 
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