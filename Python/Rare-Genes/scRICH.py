import gc
import os
import scanpy as sc
import pandas as pd
import anndata as ad
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import rc_context
import torch
from Rare_Cell_Potential_Mark_Gene_Permutation_Test.Rcpmg import rcpmg
from Rare_Cell_Potential_Mark_Gene_Permutation_Test.get_background_full import get_background_full
import scipy.stats as stats
from scipy.stats import fisher_exact
import seaborn as sns
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score, recall_score
import scipy.sparse as sp
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from collections import Counter

from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix
)
import numpy as np
import pandas as pd
import os

warnings.filterwarnings("ignore")

def merge_cluster(old_cluster, new_cluster, max_number=None):
    if not isinstance(old_cluster, pd.Series):
        old_cluster = pd.Series(old_cluster)
    if not isinstance(new_cluster, pd.Series):
        new_cluster = pd.Series(new_cluster)

    if max_number is None:
        cluster_final = old_cluster.copy()
        cluster_final.update(new_cluster)
        return cluster_final
    else:
        cluster_counts = new_cluster.value_counts()
        cluster_small = cluster_counts[cluster_counts < max_number].index
        if len(cluster_small) > 0:
            print(f"Clusters with number of cells below {max_number} in the new partition: {len(cluster_small)}")
            cluster_final = old_cluster.copy()
            for cluster in cluster_small:
                cluster_final[new_cluster == cluster] = f"rare_{cluster}"
            return cluster_final
        else:
            print(f"There are not clusters with number of cells below {max_number}")
            return old_cluster

if __name__ == "__main__":
    dataset_list = ['Quake_10x_Lung', 'Plasschaert', 'Tosches_turtle', 'Hochane', 'Han', 'Young']
    for dataset in dataset_list:
        dataset_name = dataset
        data_path = f'/data/users/feizi/ACM-GNN-main/ACM-Pytorch/output/{dataset_name}.csv'
        cell_embedding_path = f'/data/users/feizi/ACM-GNN-main/ACM-Pytorch/output/{dataset_name}/{dataset_name}_k.csv'
        write_path = f'/data/users/feizi/ACM-GNN-main/ACM-Pytorch/output/{dataset_name}_Pair'
        os.makedirs(write_path, exist_ok=True)
        result_file_path = os.path.join(write_path, '0.03_analysis_results.csv')
        F1_result_file_path = os.path.join(write_path, 'F1_results.csv')
        sorted_gene_path = os.path.join(write_path, 'sorted_genes_by_p_value.csv')
        adata_write_path = os.path.join(write_path, f'{dataset_name}_adata.h5ad')
        best_leiden_resolution = 1.0
        p_value_1 = 0.03
        p_value_2 = 0.03


        try:
            data = pd.read_csv(data_path, index_col=0)
            cell_embedding = pd.read_csv(cell_embedding_path)
            print(f"成功加载RNA的数据, 文件路径如下:")
            print(f"  - 数据文件: {data_path}")
            print(f"  - 嵌入文件: {cell_embedding_path}")
        except Exception as e:
            print(f"加载RNA的文件失败。错误信息: {e}")

        label = pd.factorize(data.iloc[:, -1])[0]
        X_data = data.iloc[:, :-1]
        adata = ad.AnnData(X=X_data.values, obs=pd.DataFrame(index=X_data.index), var=pd.DataFrame(index=X_data.columns))
        adata.obs['label'] = label

        total_cells = adata.n_obs 
        threshold = total_cells * p_value_1 
        label_counts = adata.obs['label'].value_counts()
        rare_labels = label_counts[label_counts < threshold].index

        adata.obs['rare_label'] = adata.obs['label'].apply(lambda x: 1 if x in rare_labels else 0)
        adata.obsm["cell_embedding"] = cell_embedding.values
        adata.var_names_make_unique()
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        adata.raw = adata

        adata_ciara_cluster = adata.copy()
        adata_cirar = adata.copy()
        adata_rare = adata.copy()
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution=best_leiden_resolution)
        adata.obs['initial_clusters'] = (adata.obs['leiden'].astype(int)+1).astype(str)

        cell_embedding = adata.obsm["cell_embedding"]
        adata_cirar_neighbors = sc.AnnData(cell_embedding)
        sc.pp.neighbors(adata_cirar_neighbors)
        knn_matrix = adata_cirar_neighbors.obsp["connectivities"].toarray()
        adata_cirar.obsp["connectivities"] = adata_cirar_neighbors.obsp["connectivities"]
        adata_cirar.uns["neighbors"] = adata_cirar_neighbors.uns["neighbors"]
        get_background_full(adata_cirar, threshold=1, n_cells_low=2, n_cells_high=5)
        rcpmg(adata_cirar, n_cores=20, p_value=0.001, approximation=True, local_region=1, n_permutations=1500)
        sorted_genes = adata_cirar.var.sort_values(by="RCPMG_p_value")
        sorted_genes.to_csv(sorted_gene_path, na_rep='NaN')
        significant_genes = adata_cirar.var.index[adata_cirar.var["RCPMG_p_value"] < 1].tolist()
        print("潜在稀有细胞标记基因个数：",len(significant_genes))
        top_markers = adata_cirar.var.nsmallest(4, ["RCPMG_p_value"])
        sc.tl.umap(adata_cirar, random_state=42)

        sc.pp.highly_variable_genes(adata, n_top_genes=2000, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]
        combined_genes = list(set(significant_genes + adata.var[adata.var.highly_variable].index.tolist()))
        adata_ciara_cluster = adata_ciara_cluster[:, combined_genes]
        sc.tl.pca(adata_ciara_cluster, svd_solver='arpack')
        sc.pp.neighbors(adata_ciara_cluster)
        sc.tl.leiden(adata_ciara_cluster, resolution=best_leiden_resolution)
        sc.tl.umap(adata_ciara_cluster, random_state=42)
        adata.obs['ciara_clusters'] = adata_ciara_cluster.obs['leiden']

        merged_clusters = merge_cluster(adata.obs['initial_clusters'].astype(object), adata.obs['ciara_clusters'].astype(object),max_number=round(adata.n_obs*p_value_2))
        adata.obs['final_clusters'] = merged_clusters
        adata.obs['pred_label'] = adata.obs['final_clusters'].apply(lambda x: 1 if 'rare' in str(x) else 0)

        rare_clusters = adata.obs['final_clusters'][adata.obs['final_clusters'].str.contains('rare_')]
        unique_rare_clusters = rare_clusters.unique()
        rare_cluster_mapping = {cluster: idx + 1 for idx, cluster in enumerate(sorted(unique_rare_clusters))}
        adata_rare.obs['final_clusters'] = adata.obs['final_clusters'].apply(lambda x: rare_cluster_mapping.get(x, 0))
        print("pred_label:",adata.obs['pred_label'].value_counts())

        cluster_sizes = adata.obs['final_clusters'].value_counts()
        valid_clusters = cluster_sizes[cluster_sizes > 1].index
        adata_filtered = adata[adata.obs['final_clusters'].isin(valid_clusters), :]

        true_label = adata_filtered.obs['rare_label'].to_numpy().astype(str)
        pred_label = adata_filtered.obs['pred_label'].to_numpy().astype(str)
        # adata_filtered.write(adata_write_path)

        y_true = true_label.astype(int)
        y_pred = pred_label.astype(int)

        # --- 识别稀有类数量 ---
        rare_label_set = set(rare_labels)
        num_detected_rare_types = 0
        for rare_type in rare_label_set:
            indices = adata_filtered.obs.index[adata_filtered.obs['label'] == rare_type]
            if len(indices) == 0:
                continue
            hits = adata_filtered.obs.loc[indices, 'pred_label'].sum()
            if hits / len(indices) >= 0.3:
                num_detected_rare_types += 1

        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        gmean = np.sqrt(rec * specificity)

        print(f"识别出的稀有类数量：{num_detected_rare_types}/{len(rare_labels)}")
        print(f"F1 Score : {f1:.4f}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"MCC      : {mcc:.4f}")
        print(f"Kappa    : {kappa:.4f}")
        print(f"G-mean   : {gmean:.4f}")

        results_df = pd.DataFrame({
            'F1_score': [f1],
            'Accuracy': [acc],
            'Recall': [rec],
            'MCC': [mcc],
            'Kappa': [kappa],
            'G-mean': [gmean],
            'Rare_Types_Detected': [num_detected_rare_types],
            'Total_Rare_Types': [len(rare_labels)]
        })

        metrics_output_path = os.path.join(write_path, 'F1_detailed_results.csv')
        results_df.to_csv(metrics_output_path, index=False)
        print(f"详细结果保存到 {metrics_output_path}")
