import os
import sys
import logging
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ===== 配置日志系统 =====
def setup_logging():
    """配置详细的日志输出"""
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 控制台输出格式
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()


from grid_search import PARAM_GRID, DLPFC_SAMPLES, grid_search

# ===== 打印开始信息 =====
logger.info("="*80)
logger.info("GRID SEARCH FOR HYPERPARAMETER OPTIMIZATION")
logger.info("="*80)
logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info("")

logger.info("Search Space:")
for param, values in PARAM_GRID.items():
    logger.info(f"  {param}: {values}")
logger.info(f"Total combinations: {3**5} = 243")
logger.info("")

logger.info("Datasets:")
for sample, n_clusters in DLPFC_SAMPLES.items():
    logger.info(f"  {sample}: {n_clusters} clusters")
logger.info("")

# ===== 运行网格搜索 =====
try:
    logger.info("Starting grid search...")
    
    # 方案1: 完整搜索
    # best_params, results_df = grid_search(
    #     param_grid=PARAM_GRID,
    #     samples_dict=DLPFC_SAMPLES,
    #     save_results=True
    # )
    
    # 方案2: 快速测试 (30个随机组合)
    best_params, results_df = grid_search(
        param_grid=PARAM_GRID,
        samples_dict=DLPFC_SAMPLES,
        max_combinations=120,
        save_results=True
    )
    
    # ===== 打印最终结果 =====
    logger.info("")
    logger.info("="*80)
    logger.info("GRID SEARCH COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best mean ARI: {results_df.iloc[0]['mean_ari']:.4f}")
    logger.info("")
    
    logger.info("Top 5 combinations:")
    top5 = results_df.head(5)[['combination_id', 'mean_ari', 'std_ari', 
                                'gamma', 'kappa', 'graph_corr', 'dropout', 
                                'residual_weight']]
    logger.info("\n" + top5.to_string())
    
    logger.info("")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Results saved to: grid_search_results_final.csv")
    logger.info("="*80)
    
except Exception as e:
    logger.error("="*80)
    logger.error("GRID SEARCH FAILED!")
    logger.error("="*80)
    logger.error(f"Error: {str(e)}")
    logger.error("", exc_info=True)
    sys.exit(1)