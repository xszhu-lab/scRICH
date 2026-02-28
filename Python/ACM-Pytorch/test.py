import torch
import pandas as pd
import scanpy as sc

# # 加载张量数据
# data = torch.load('ACM-GNN-main/ACM-Pytorch/features2.pt')

# # 查看数据内容
# print(data)

# # 检查数据的形状
# print(data.shape)

# # 将张量数据转换为 DataFrame
# data1 = pd.DataFrame(data.numpy())

# # 打印 DataFrame 的形状
# print(data1.shape)
adata = sc.read_h5ad('/home/feizi/ACM-GNN-main/ACM-Pytorch/output/mouse_all/mouse_all/mouse_all_adata.h5ad')

print(adata.obs['final_clusters'].unique())

label = adata.obs['final_clusters']

label.to_csv('/home/feizi/ACM-GNN-main/ACM-Pytorch/output/mouse_all/mouse_all/mouse_all_adata.csv', index=True)


