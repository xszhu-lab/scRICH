import os
import pandas as pd
import scanpy as sc
from Rare_Cell_Potential_Mark_Gene_Permutation_Test.Rcpmg import rcpmg
from Rare_Cell_Potential_Mark_Gene_Permutation_Test.get_background_full import get_background_full
# from Rcpmg import *
# from get_background_full import *

def merge_predicted_labels(y_true, y_pred):
    """
    合并预测标签，根据每个预测标签类别中主导的真实类别重新分配预测标签。
    
    Args:
        y_true (np.ndarray): n*1 的真实标签数组。
        y_pred (np.ndarray): n*1 的预测标签数组。
    
    Returns:
        np.ndarray: 合并后的预测标签数组。
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 存储每个预测类别对应的主导真实类别
    pred_to_true_map = {}

    # 遍历每个预测类别
    for pred_label in np.unique(y_pred):
        # 获取当前预测类别对应的真实标签
        true_labels_in_cluster = y_true[y_pred == pred_label]
        
        # 找出主导的真实类别
        most_common_label = Counter(true_labels_in_cluster).most_common(1)[0][0]
        
        # 映射当前预测类别到主导的真实类别
        pred_to_true_map[pred_label] = most_common_label

    # 用主导的真实类别替换预测标签
    merged_pred_labels = np.array([pred_to_true_map[label] for label in y_pred])

    return merged_pred_labels

def purity_score(y_true, y_pred):
    """Purity score

    Args:
        y_true (np.ndarray): n*1 matrix, true labels
        y_pred (np.ndarray): n*1 matrix, predicted clusters

    Returns:
        float: Purity score
    """
    # Create a matrix to store the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)

    # Sort the labels
    # Some labels might be missing, e.g., a set {0,2} where 1 is missing
    # First, find the unique labels and then map them to an ordered set
    # E.g., {0,2} should be mapped to {0,1}
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    y_true = np.array(y_true, dtype='int64')

    # Update the unique labels
    labels = np.unique(y_true)

    # Set the number of bins to n_classes + 2 so that we can compute the actual
    # class occurrences between two consecutive bins
    # The larger bin is excluded: [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most frequent label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    y_true = np.array(y_true, dtype='int8')
    y_voted_labels = np.array(y_voted_labels, dtype='int8')
    return accuracy_score(y_true, y_voted_labels), y_true

def Entropy(pred_label, true_label):
    e = 0
    for k in set(pred_label):
        en = 0
        pred_k = Counter(pred_label)[k]
        index_pred_k = pred_label == k
        for j in set(true_label):
            true_j = Counter(true_label)[j]
            intersection_kj = (true_label[index_pred_k] == j).sum()
            p = np.array(intersection_kj) / np.array(pred_k)
            if p != 0:
                en += np.log(p) * p
        e = e + en * pred_k / true_label.shape[0]
    return abs(e)

def preprocess_rna_data(data_path, cell_embedding_path, p_value_1=0.01):
    """加载数据并预处理, 返回AnnData对象"""
    try:
        # 加载数据
        data = pd.read_csv(data_path, index_col=0)
        cell_embedding = pd.read_csv(cell_embedding_path)
    except Exception as e:
        print(f"加载文件失败。错误信息: {e}")
        return None
    
    # 转换为AnnData格式
    # print("data : ", data)
    # print("cell embedding : ", cell_embedding)
    label = pd.factorize(data['label'])[0]
    X_data = data.iloc[:, :-1]
    adata = sc.AnnData(X=X_data.values, obs=pd.DataFrame(index=X_data.index), var=pd.DataFrame(index=X_data.columns))
    adata.obs['label'] = label
    adata.obsm["cell_embedding"] = cell_embedding.values
    
    # 标记稀有标签
    total_cells = adata.n_obs
    threshold = total_cells * p_value_1
    label_counts = adata.obs['label'].value_counts()
    rare_labels = label_counts[label_counts < threshold].index
    adata.obs['rare_label'] = adata.obs['label'].apply(lambda x: -1 if x in rare_labels else x)
    
    return adata

def optimize_clustering(adata, resolution_start=1.0, resolution_step=0.2, resolution_min=0.2, resolution_max=2.0):
    """优化Leiden聚类的分辨率"""
    def optimize_leiden_resolution(adata, true_label_column='label', initial_resolution=resolution_start, resolution_step=resolution_step, max_resolution=3.0, min_resolution=0.2, tolerance=1):
        """
        优化Leiden聚类的分辨率, 直到聚类数最接近真实标签数。

        参数：
        - adata: AnnData对象
        - true_label_column: 存储真实标签的列名(默认为 'label')
        - initial_resolution: 初始分辨率(默认为1.6)
        - resolution_step: 分辨率调整步长(默认为0.1)
        - max_resolution: 最大分辨率限制(默认为3.0)
        - min_resolution: 最小分辨率限制(默认为0.2)
        - tolerance: 允许的差异容忍值(默认为0.01)

        返回：
        - 最优分辨率值
        """
        # 获取真实标签数
        true_label_count = len(adata.obs[true_label_column].unique())
        print("true_label_count", true_label_count)

        resolution = initial_resolution
        leiden_label_count = 0
        best_resolution = resolution
        min_diff = float('inf')
        previous_diff = float('inf')

        while True:
            # 执行Leiden聚类
            print("当前resolution:", resolution)
            sc.tl.leiden(adata, resolution=resolution)
            leiden_label_count = len(adata.obs['leiden'].unique())

            # 计算Leiden聚类标签数与真实标签数的差异
            diff = abs(leiden_label_count - true_label_count)

            # 如果当前差异更小，更新最优分辨率
            if diff < min_diff:
                min_diff = diff
                best_resolution = resolution

            # 判断是否继续调整分辨率
            if leiden_label_count < true_label_count:
                resolution += resolution_step
            else:
                resolution -= resolution_step

            # 确保分辨率不超出范围
            if resolution > max_resolution:
                resolution = max_resolution
            elif resolution < min_resolution:
                resolution = min_resolution

            # 检查是否需要停止
            if abs(previous_diff - diff) < tolerance:  # 差异变化小于容忍值，提前停止
                print("差异变化小于容忍值，停止优化。")
                break
            
            # 如果分辨率不再变化，则退出
            if leiden_label_count == true_label_count or resolution == max_resolution or resolution == min_resolution:
                break

            previous_diff = diff

        print(f"Optimal resolution: {best_resolution}")
        return best_resolution

    best_resolution = optimize_leiden_resolution(adata, resolution_start, resolution_step, resolution_min, resolution_max)
    return best_resolution

def ciara_analysis(adata, threshold, n_cells_low, n_cells_high, p_value, local_region, n_permutations):
    """进行CIARA分析"""
    # 计算背景
    get_background_full(adata, threshold=threshold, n_cells_low=n_cells_low, n_cells_high=n_cells_high)
    
    # 使用RCMPG方法筛选基因
    rcpmg(adata, n_cores=10, p_value=p_value, approximation=True, local_region=local_region, n_permutations=n_permutations)
    
    # 筛选显著基因
    adata.var["RCPMG_p_value"] = adata.var["RCPMG_p_value"].fillna(1)
    significant_genes = adata.var.index[adata.var["RCPMG_p_value"] < p_value].tolist()
    
    return adata, significant_genes

def merge_clusters(adata, significant_genes, best_resolution, p_value_2):
    """
    根据显著基因重新聚类并合并初始聚类和最终聚类。

    参数:
        adata (AnnData): 包含单细胞数据的AnnData对象。
        significant_genes (list): 显著基因列表。
        best_resolution (float): 用于Leiden聚类的分辨率参数。
        p_value_2 (float): 用于定义小类别的阈值比例。

    返回:
        AnnData: 更新后的AnnData对象, 包含最终聚类和预测标签。
    """
    # 基于显著基因提取子集
    adata_ciara_cluster = adata[:, significant_genes]
    # PCA降维
    sc.tl.pca(adata_ciara_cluster, svd_solver='arpack')
    # 构建邻居图
    sc.pp.neighbors(adata_ciara_cluster)
    # Leiden聚类
    sc.tl.leiden(adata_ciara_cluster, resolution=best_resolution)
    
    # 调用提供的merge_cluster函数
    merged_clusters = merge_cluster(
        adata.obs['initial_clusters'],  # 初始聚类
        adata_ciara_cluster.obs['leiden'],  # Leiden聚类结果
        max_number=round(adata.n_obs * p_value_2)  # 小类别的最大细胞数
    )
    
    # 将合并的结果存入adata的obs中
    adata.obs['final_clusters'] = merged_clusters
    # 预测标签，'rare'标记为-1
    adata.obs['pred_label'] = adata.obs['final_clusters'].apply(lambda x: -1 if 'rare' in str(x) else x)
    
    return adata

def pipeline(data_path, cell_embedding_path, write_path, p_value_1=0.01, p_value_2=0.01):
    """完整的RNA数据分析流程"""
    adata = preprocess_rna_data(data_path, cell_embedding_path, p_value_1)
    if adata is None:
        return
    
    # 计算高变基因和初始聚类
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata)
    
    # 优化分辨率并初始聚类
    best_resolution = optimize_clustering(adata)
    sc.tl.leiden(adata, resolution=best_resolution)
    adata.obs['initial_clusters'] = (adata.obs['leiden'].astype(int) + 1).astype(str)
    
    # CIARA分析
    adata, significant_genes = ciara_analysis(adata, threshold=1, n_cells_low=2, n_cells_high=5, p_value=0.001, local_region=1, n_permutations=2000)
    
    # 保存排序基因
    sorted_gene_path = os.path.join(write_path, 'sorted_genes_by_p_value.csv')
    adata.var.sort_values(by="RCPMG_p_value").to_csv(sorted_gene_path, na_rep='NaN')
    
    # 聚类合并
    adata = merge_clusters(adata, significant_genes, best_resolution, p_value_2)
    
    # 返回最终结果
    return adata
