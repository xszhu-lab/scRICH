import torch
import pandas as pd

# 加载 .pt 文件
dataset_name = 'Kumar'

data = torch.load(f'output/{dataset_name}/{dataset_name}.pt', map_location=torch.device('cpu'))

# 假设数据是字典，可以选择需要的数据
# 如果数据是张量，也可以直接转换为 DataFrame
if isinstance(data, dict):
    for key, value in data.items():
        # 检查张量是否需要梯度跟踪，如果是，则先调用 detach()
        if value.requires_grad:
            value = value.detach()
        df = pd.DataFrame(value.numpy())
        df.to_csv(f'{key}.csv', index=False)
elif isinstance(data, torch.Tensor):
    # 同样检查张量是否需要梯度跟踪
    if data.requires_grad:
        data = data.detach()
    df = pd.DataFrame(data.numpy())
    df.to_csv(f'output/{dataset_name}/{dataset_name}_k.csv', index=False)
else:
    print("无法直接转换为CSV格式的数据类型。")

print("转换完成")