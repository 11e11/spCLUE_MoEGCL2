import scanpy as sc 
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

import scipy.sparse as sp 
import warnings

warnings.filterwarnings("ignore")

import os
import ctypes
import sys

# 1. 先设置 R_HOME
os.environ["R_HOME"] = "/home/pxy/miniconda3/envs/r40/lib/R"

# 2. 【核心黑科技】手动加载 R 的动态库
# 这步操作等同于在终端里设置 LD_LIBRARY_PATH，专门解决 VS Code 找不到库的问题
try:
    # 这是 R 的核心库路径
    libR_path = "/home/pxy/miniconda3/envs/r40/lib/R/lib/libR.so"
    # 强制加载进内存
    ctypes.CDLL(libR_path, mode=ctypes.RTLD_GLOBAL)
    print("✅ 成功强制加载 libR.so")
except OSError as e:
    print(f"❌ 加载失败: {e}")

# 3. 然后再导入其他包
sys.path.append("..") 

import spCLUE
import rpy2.robjects as robjects
print("R 环境路径:", robjects.r['R.home']()[0])

spCLUE.fix_seed(0)

# 定义DLPFC数据集的12个切片ID
slice_ids = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]

# 用于存储每个切片的ARI结果
ari_results_kmeans = []
ari_results_mclust = []

# 数据路径（请根据实际情况确认路径是否正确）
data_dir = '/home/pxy/home/pxy/data/DLPFC/st/'

# 【新增】创建保存图片的文件夹
figures_dir = "figures_test"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"Created directory: {figures_dir}")

print(f"Start processing {len(slice_ids)} slices...")

for sample_name in slice_ids:
    print(f"\n{'='*20} Processing Sample: {sample_name} {'='*20}")
    
    # 1. 设置簇的数量 (根据DLPFC数据集的已知Ground Truth)
    # 151669-151672 通常只有5层，其他切片为7层
    if sample_name in ["151669", "151670", "151671", "151672"]:
        n_clusters = 5
    else:
        n_clusters = 7
    
    try:
        # 2. 加载数据
        # 使用 read_visium 加载数据，路径拼接逻辑参考原文件
        adata = sc.read_visium(data_dir + sample_name)
        adata.var_names_make_unique()
        
        # 加载元数据 (Ground Truth)
        meta = pd.read_csv(data_dir + sample_name + "/metadata.tsv", sep="\t")
        meta = meta.set_index("barcode")
        adata.obs["Region"] = meta.loc[adata.obs_names, "layer_guess_reordered"]
        
        # 3. 数据预处理与构图
        # 原文件 Cell 6 的逻辑
        adata = spCLUE.preprocess(adata)
        adata.obsm["X_pca"] = PCA(n_components=200, random_state=0).fit_transform(adata.X)
        g_spatia = spCLUE.prepare_graph(adata, "spatial",n_neighbors=6)
        g_expr = spCLUE.prepare_graph(adata, "expr", metric="euclidean",n_neighbors=12)
        graph_dict = {"spatial": g_spatia, "expr":g_expr}
                
        
    #     model = spCLUE.spCLUE_TwoStage(
    #     adata.obsm["X_pca"], 
    #     graph_dict, 
    #     n_clusters=n_clusters,
    #     pretrain_epochs=100,   # 预训练200轮
    #     finetune_epochs=100,   # 训练300轮
    #     gamma=0.5,             # 重构损失权重
    #     beta=0.0,              # 聚类损失权重=0 (关键!)
    #     kappa=0.5,             # 对比损失权重
    #     dim_hidden=32,
    #     freeze_encoder=True,   # 冻结预训练编码器
    #     graph_corr=0.5,
    #     dropout=0.5,
    #     residual_weight=0.3
    # )
        
        model = spCLUE.spCLUE_TwoStage(
            adata.obsm["X_pca"], 
            # input_data,
            graph_dict, 
            n_clusters=n_clusters,
            dim_input=200,
            pretrain_epochs=100,   # 预训练200轮
            finetune_epochs=100,   # 训练300轮
            gamma=0.0,             # 重构损失权重
            beta=0.0,              # 聚类损失权重=0 (关键!)
            kappa=2.0,             # 对比损失权重
            dim_hidden=32,
            freeze_encoder=True,   # 冻结预训练编码器
            graph_corr=0.4,
            dropout=0.1,
            residual_weight=0.1
        )
        pred, embed, gated_weights = model.train()
        # 5. 聚类
        # 原文件 Cell 10 的逻辑
        # ========== 聚类 ==========
        adata.obsm["spCLUE_twostage"] = embed
        spCLUE.clustering(adata, n_clusters, key="spCLUE_twostage", refinement=True,cluster_methods='kmeans')

        # ========== 评估 ==========
        adata_filtered = adata[adata.obs.Region.notna()]
        ARI_kmeans = adjusted_rand_score(adata_filtered.obs["Region"], 
                                adata_filtered.obs["kmeans_refined"])
        print(f"\nFinal Kmeans ARI on {sample_name}: {ARI_kmeans:.8f}")
        ari_results_kmeans.append(ARI_kmeans)

        # ========== 聚类 ==========
        adata.obsm["spCLUE_twostage"] = embed
        spCLUE.clustering(adata, n_clusters, key="spCLUE_twostage", refinement=True)

        # ========== 评估 ==========
        adata_filtered = adata[adata.obs.Region.notna()]
        ARI_mclust = adjusted_rand_score(adata_filtered.obs["Region"], 
                                adata_filtered.obs["mclust_refined"])
        # print(f"\nFinal Mclust ARI on {sample_name}: {ARI:.4f}")
        
        # 6. 计算 ARI
        # 原文件 Cell 12 的逻辑
        # 过滤掉 Ground Truth 为 NA 的区域
        # adata_valid = adata[adata.obs.Region.notna()]
        # ARI = adjusted_rand_score(adata_valid.obs["Region"], adata_valid.obs["mclust_refined"])
        
        print(f"Sample {sample_name} ARI: {ARI_mclust:.8f}")
        ari_results_mclust.append(ARI_mclust)

        # 绘图：show=False 防止直接显示，便于后续保存
        # adata.obs["spCLUE"] = adata.obs["mclust_refined"]
        # sc.pl.spatial(
        #     adata, 
        #     color=["Region", "spCLUE"], 
        #     title=["Manual Annotation", f"spCLUE (ARI={round(ARI, 2)})"],
        #     show=False 
        # )
        
        # # 保存路径
        # save_path = os.path.join(figures_dir, f"{sample_name}.png")
        
        # # 保存图片 (bbox_inches='tight' 去除多余白边, dpi=300 保证清晰度)
        # plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # # 关闭当前图形，释放内存 (在循环中非常重要，否则内存会爆)
        # plt.close()
        
        # print(f"Figure saved to: {save_path}")
        
    except Exception as e:
        print(f"Error processing sample {sample_name}: {e}")

# 7. 输出最终统计结果
print(f"\n{'='*20} Final Results {'='*20}")
if ari_results_kmeans:
    mean_ari = np.mean(ari_results_kmeans)
    median_ari = np.median(ari_results_kmeans)
    print(f"ARI per slice: {[round(x, 5) for x in ari_results_kmeans]}")
    print(f"Mean ARI: {mean_ari:.4f}")
    print(f"Median ARI: {median_ari:.4f}")
else:
    print("No ARI results collected.")
    print(f"\n{'='*20} Final Results {'='*20}")
if ari_results_mclust:
    mean_ari = np.mean(ari_results_mclust)
    median_ari = np.median(ari_results_mclust)
    print(f"ARI per slice: {[round(x, 5) for x in ari_results_mclust]}")
    print(f"Mean ARI: {mean_ari:.4f}")
    print(f"Median ARI: {median_ari:.4f}")
else:
    print("No ARI results collected.")