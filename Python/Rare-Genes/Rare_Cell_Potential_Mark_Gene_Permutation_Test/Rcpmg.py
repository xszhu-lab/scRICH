import torch
import numpy as np
import scipy.sparse as sp
import multiprocessing
from functools import partial
from tqdm import tqdm
import pandas as pd
import scipy
import warnings
from statsmodels.stats.multitest import multipletests
gene_expressions_g = None
knn_matrix_g = None


def perform_permutation_test(observed_sum, gene_expression, nn_cells, n_permutations, nn_weights):
    permuted_sums = []

    for _ in range(n_permutations):
        permuted_expression = np.random.permutation(gene_expression)
        # permuted_sum = np.sum(permuted_expression[nn_cells] > 0) / len(nn_cells)
        if len(nn_weights)!=0:
            permuted_sum = np.average(permuted_expression[nn_cells], weights=nn_weights)
        else:
            permuted_sum = 0
        # permuted_sum = np.average(permuted_expression[nn_cells], weights=nn_weights)
        permuted_sums.append(permuted_sum)

    permuted_sums = np.array(permuted_sums)
    p_value = np.sum(permuted_sums >= observed_sum) / n_permutations

    return p_value


def ciara_gene(gene_idx, gene_expressions, knn_matrix, p_value, local_region, approximation, n_permutations):
    gene_expression = gene_expressions[gene_idx]
    binary_expression = gene_expression > np.median(gene_expression)

    if approximation:
        knn_subset = np.nonzero(binary_expression)[0]
    else:
        knn_subset = range(knn_matrix.shape[0])

    #保存每个细胞的P值以及每个细胞的邻域大小
    p_values_nn = []
    weights = []

    for cell in knn_subset:
        nn_cells = np.nonzero(knn_matrix[cell, :])[0]
        #该观测值是该细胞邻域中该基因表达的占比
        # observed_sum = np.sum(gene_expression[nn_cells] > 0) / len(nn_cells)
        #该观测值是该细胞邻域中该基因表达的均值
        # observed_sum = np.mean(gene_expression[nn_cells])
        #根据权重举证计算加权观测值
        nn_weights = knn_matrix[cell, nn_cells]
        if len(nn_weights)!=0:
            observed_sum = np.average(gene_expression[nn_cells], weights=nn_weights)
        else:
            observed_sum = 0
        p_value_nn = perform_permutation_test(observed_sum, gene_expression, nn_cells, n_permutations, nn_weights)
        p_values_nn.append(p_value_nn)
        weights.append(len(nn_cells))

    p_values_nn = np.array(p_values_nn, dtype=float)


    if np.sum(np.array(p_values_nn) < p_value) >= local_region:
        # p_value_gene = np.average(p_values_nn)
        p_value_gene = np.average(p_values_nn, weights=weights)
    else:
        p_value_gene = 1


    return p_value_gene


def rcpmg(norm_adata, n_cores, p_value, local_region, approximation, n_permutations):
    #多线程
    if multiprocessing.get_start_method(allow_none=True) != 'fork':
        multiprocessing.set_start_method("spawn", force=True)

    background = norm_adata.X[:, norm_adata.var["RCPMG_background"]]
    #判断是否是稀疏矩阵，如果是稀疏矩阵则转换成密集矩阵
    if sp.issparse(background):
        background = background.toarray()
    #
    gene_expressions = [background[:, i].flatten() for i in range(np.shape(background)[1])]
    # csv_adj_file = 'data/adj_100.csv'
    # adj = pd.read_csv(csv_adj_file, index_col=0)
    # adj = adj.iloc[:3587, :3587]
    # adj_sparse_matrix = sp.csr_matrix(adj.values)
    # knn_matrix = adj_sparse_matrix.toarray()
    knn_matrix = norm_adata.obsp["connectivities"].toarray()
    # knn_distances_matrix = norm_adata.obsp["distances"].toarray()

    chunksize, extra = divmod(len(gene_expressions), 4 * n_cores)
    if extra:
        chunksize += 1
    print("\n## Running on " + str(n_cores) + " cores with a chunksize of " + str(chunksize))

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=len(gene_expressions))

    def update_progress(*a):
        progress_bar.update()

    with multiprocessing.Pool(n_cores) as pool:
        temp = partial(ciara_gene, gene_expressions=gene_expressions, knn_matrix=knn_matrix, p_value=p_value,
                       local_region=local_region, approximation=approximation, n_permutations=n_permutations)
        # results = pool.map(func=temp, iterable=range(len(gene_expressions)), chunksize=chunksize)
        results = [pool.apply_async(temp, args=(i,), callback=update_progress) for i in range(len(gene_expressions))]
        results = [r.get() for r in results]

    progress_bar.close()


    p_values_output = [np.NAN for _ in range(len(norm_adata.var_names))]
    for index, gene_pos in enumerate(np.where(norm_adata.var["RCPMG_background"])[0]):
        p_values_output[gene_pos] = results[index]

    norm_adata.var.drop(columns="RCPMG_p_value", inplace=True, errors='ignore')
    norm_adata.var.insert(0, "RCPMG_p_value", p_values_output)

    print('\n---- Finished successfully! ----')
    return

