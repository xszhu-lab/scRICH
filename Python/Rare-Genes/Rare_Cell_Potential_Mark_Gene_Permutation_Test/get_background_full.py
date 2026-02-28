import numpy as np

def get_background_full(norm_adata, threshold, n_cells_low, n_cells_high):

    thr_per_gene = np.sum(norm_adata.X > threshold, axis=0).flat
    genes_filter = np.logical_and(thr_per_gene >= n_cells_low, thr_per_gene <= n_cells_high)
    print("Background genes: " + str(np.sum(genes_filter)))

    norm_adata.var.drop(columns="RCPMG_background", inplace=True, errors='ignore')
    norm_adata.var.insert(0, "RCPMG_background", genes_filter.tolist())

    return

def integration_background_gene(adata, sp_adata):
    RCPMG_background_bool = adata.var["RCPMG_background"]
    RCPMG_background_names = RCPMG_background_bool[RCPMG_background_bool == True].index.tolist()
    # 获取基因名
    sp_adata_gene_names = sp_adata.var_names.tolist()
    # 取交集
    intersection_genes = list(set(RCPMG_background_names).intersection(sp_adata_gene_names))

    # Update adata.var["RCPMG_background"]
    adata.var.drop(columns="RCPMG_background", inplace=True, errors='ignore')
    # 重新创建 RCPMG_background 列，并初始化为 False
    adata.var["RCPMG_background"] = False
    # 将交集内的基因名设为 True
    adata.var.loc[intersection_genes, "RCPMG_background"] = True

    sp_adata = sp_adata[:, intersection_genes]

    return adata, sp_adata