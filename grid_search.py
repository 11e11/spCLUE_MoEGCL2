from datetime import datetime,timedelta
import itertools
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import scanpy as sc
from sklearn.decomposition import PCA
import spCLUE
import warnings
warnings.filterwarnings("ignore")

# ===== 超参数搜索空间 =====
PARAM_GRID = {
    'gamma': [0.5, 1.0, 2.0],              # 重构损失权重
    'kappa': [0.5, 1.0, 2.0],              # 对比损失权重
    'graph_corr': [0.3, 0.4, 0.5],         # 边dropout率
    'dropout': [0.1, 0.3, 0.5],            # 特征dropout率
    'residual_weight': [0.1, 0.2, 0.3],    # 残差连接权重
}

# DLPFC数据集配置
DLPFC_SAMPLES = {
    '151669': 5, '151670': 5, '151671': 5, '151672': 5,  # n_clusters=5
    '151673': 7, '151674': 7, '151675': 7, '151676': 7,  # n_clusters=7
    '151507': 7, '151508': 7, '151509': 7, '151510': 7,
}

DATA_DIR = '/home/pxy/home/pxy/data/DLPFC/st/'

def train_single_sample(sample_name, n_clusters, params, data_dir=DATA_DIR, verbose=False):
    """
    在单个DLPFC样本上训练模型
    
    Args:
        sample_name: 样本名称 (如'151671')
        n_clusters: 聚类数
        params: 超参数字典
        data_dir: 数据目录
        verbose: 是否打印详细信息
    
    Returns:
        ari: 该样本的ARI分数
    """
    import scanpy as sc 
    import pandas as pd 
    import numpy as np 

    import matplotlib.pyplot as plt 
    import seaborn as sns 

    from sklearn.metrics import adjusted_rand_score
    from sklearn.decomposition import PCA

    import scipy.sparse as sp 
    import warnings
    import torch

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
    
    try:
        # ===== 加载数据 =====
        adata = sc.read_visium(data_dir + sample_name)
        adata.var_names_make_unique()
        
        # 加载标签
        meta = pd.read_csv(
            data_dir + sample_name + "/metadata.tsv",
            sep="\t"
        )
        meta = meta.set_index("barcode")
        adata.obs["Region"] = meta.loc[adata.obs_names, "layer_guess_reordered"]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {sample_name} (n_clusters={n_clusters})")
            print(f"{'='*60}")
            print(f"Data shape: {adata.shape}")
            print(f"NA spots: {adata.obs['Region'].isna().sum()}")
        
        # ===== 预处理 =====
        adata = spCLUE.preprocess(adata)
        adata.obsm["X_pca"] = PCA(n_components=200, random_state=0).fit_transform(adata.X)
        
        # 构建图
        g_spatial = spCLUE.prepare_graph(adata, "spatial")
        g_expr = spCLUE.prepare_graph(adata, "expr", metric="euclidean")
        graph_dict = {"spatial": g_spatial, "expr": g_expr}
        
        # ===== 训练模型 =====
        model = spCLUE.spCLUE_TwoStage(
            adata.obsm["X_pca"], 
            graph_dict, 
            n_clusters=n_clusters,
            pretrain_epochs=100,
            finetune_epochs=100,
            gamma=params['gamma'],
            beta=0.0,  # 固定为0
            kappa=params['kappa'],
            graph_corr=params['graph_corr'],
            dropout=params['dropout'],
            freeze_encoder=True,
            residual_weight=params['residual_weight'],  # 传递残差权重
        )
        
        pred, embed = model.train()
        
        # ===== 聚类 =====
        adata.obsm["spCLUE_twostage"] = embed
        spCLUE.clustering(adata, n_clusters, key="spCLUE_twostage", refinement=True)
        
        # ===== 评估 =====
        adata_filtered = adata[adata.obs.Region.notna()]
        ari = adjusted_rand_score(
            adata_filtered.obs["Region"], 
            adata_filtered.obs["mclust_refined"]
        )
        
        if verbose:
            print(f"✓ {sample_name} ARI: {ari:.4f}")
        
        return ari
    
    except Exception as e:
        print(f"✗ Error in {sample_name}: {str(e)}")
        return 0.0  # 失败返回0

def grid_search(param_grid, samples_dict, data_dir=DATA_DIR, 
                max_combinations=None, save_results=True):
    """网格搜索 (添加详细日志)"""
    logger = logging.getLogger()
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    all_combinations = list(itertools.product(*param_values))
    
    # 限制搜索数量
    if max_combinations and len(all_combinations) > max_combinations:
        logger.info(f"Total combinations: {len(all_combinations)}, "
                   f"limiting to {max_combinations} random samples")
        import random
        random.seed(42)
        all_combinations = random.sample(all_combinations, max_combinations)
    else:
        logger.info(f"Total combinations to search: {len(all_combinations)}")
    
    results = []
    start_time = datetime.now()
    
    # 遍历所有参数组合
    for idx, param_values in enumerate(all_combinations, 1):
        params = dict(zip(param_names, param_values))
        
        logger.info("")
        logger.info("="*80)
        logger.info(f"Combination {idx}/{len(all_combinations)}")
        logger.info(f"Parameters: {params}")
        logger.info("="*80)
        
        # 在所有12个样本上训练
        sample_aris = []
        for sample_name, n_clusters in samples_dict.items():
            logger.info(f"Processing {sample_name} (n_clusters={n_clusters})...")
            ari = train_single_sample(
                sample_name, n_clusters, params, data_dir, verbose=False
            )
            sample_aris.append(ari)
            logger.info(f"  → {sample_name} ARI: {ari:.4f}")
        
        # 计算平均ARI
        mean_ari = np.mean(sample_aris)
        std_ari = np.std(sample_aris)
        
        logger.info("-"*80)
        logger.info(f"Combination {idx} results:")
        logger.info(f"  Mean ARI: {mean_ari:.4f} ± {std_ari:.4f}")
        logger.info(f"  Min ARI:  {np.min(sample_aris):.4f}")
        logger.info(f"  Max ARI:  {np.max(sample_aris):.4f}")
        
        # 计算预计剩余时间
        elapsed = (datetime.now() - start_time).total_seconds()
        avg_time_per_comb = elapsed / idx
        remaining_comb = len(all_combinations) - idx
        eta_seconds = avg_time_per_comb * remaining_comb
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        logger.info(f"  Progress: {idx}/{len(all_combinations)} ({idx/len(all_combinations)*100:.1f}%)")
        logger.info(f"  Elapsed: {str(timedelta(seconds=int(elapsed)))}")
        logger.info(f"  ETA: {eta_str}")
        logger.info("="*80)
        
        # 保存结果
        result = {
            'combination_id': idx,
            'mean_ari': mean_ari,
            'std_ari': std_ari,
            **params,
        }
        for sample_name, ari in zip(samples_dict.keys(), sample_aris):
            result[f'ari_{sample_name}'] = ari
        
        results.append(result)
        
        # 实时保存中间结果
        if save_results:
            temp_df = pd.DataFrame(results)
            temp_df = temp_df.sort_values('mean_ari', ascending=False)
            temp_df.to_csv('grid_search_results_temp.csv', index=False)
            logger.info(f"✓ Temp results saved (current best: {temp_df.iloc[0]['mean_ari']:.4f})")
    
    # 转为DataFrame
    results_df = pd.DataFrame(results)
    
    # 按mean_ari降序排列
    results_df = results_df.sort_values('mean_ari', ascending=False)
    
    # 找到最优参数
    best_idx = results_df.iloc[0]['combination_id']
    best_params = {k: results_df.iloc[0][k] for k in param_names}
    best_mean_ari = results_df.iloc[0]['mean_ari']
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE!")
    print(f"{'='*80}")
    print(f"Best combination #{int(best_idx)}:")
    print(f"  Parameters: {best_params}")
    print(f"  Mean ARI: {best_mean_ari:.4f}")
    print(f"{'='*80}")
    
    # 保存最终结果
    if save_results:
        results_df.to_csv('grid_search_results_final.csv', index=False)
        print(f"✓ Results saved to: grid_search_results_final.csv")
    
    return best_params, results_df