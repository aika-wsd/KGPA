from torch.utils.data import Dataset

"""
比如对于下面的数据:
dataset = {
    'num_rows': 3,
    0: {'idx': 1, 'age': 25, 'income': 50000, 'label': 0},
    1: {'idx': 2, 'age': 30, 'income': 60000, 'label': 1},
    2: {'idx': 3, 'age': 22, 'income': 55000, 'label': 0}
}
使用本类处理之后的格式为:
Dataset length: 3
Data at index 0: [['age', 25], ['income', 50000]], Label: 0
Data at index 1: [['age', 30], ['income', 60000]], Label: 1
Data at index 2: [['age', 22], ['income', 55000]], Label: 0
"""

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.len = dataset.num_rows # 存储数据集中的行数，确定数据集的大小
        # 如果键不是"label"或"idx"
        # 则将该键值对作为一个子列表[key, value]添加到列表中
        # 这样做是为了分离特征数据和标签或其他不需要的信息
        self.data = [
            (
                [
                    [key, value]
                    for (key, value) in dataset[i].items()
                    if key != "label" and key != "idx"
                ]
            )
            for i in range(dataset.num_rows) # 为每行数据构建特征列表
        ]
        self.label = [dataset[i]["label"] for i in range(dataset.num_rows)]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # 返回一个包含特定索引处的特征和标签的元组
        # 便于进行数据处理和迭代
        return self.data[idx], self.label[idx]
