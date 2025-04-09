import time
import numpy as np
import torch
from flcore.clients.clientavg import clientAVG
from utils.data_utils import read_client_data

class clientFedEntPlus(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        # 为可选的 train_slow 和 send_slow 提供默认值，调用父类构造函数
        kwargs.setdefault('train_slow', False)
        kwargs.setdefault('send_slow', False)
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # 获取实际训练数据（由服务器在初始化时通过 kwargs 传入）
        train_data = kwargs.get('real_train_data', None)
        if train_data is None:
            # 若未提供真实数据列表，则自行读取（保证兼容性）
            train_data = read_client_data(self.dataset, self.id, is_train=True)
        # 统计本地标签分布：初始化计数向量
        self.label_count = np.zeros(self.num_classes, dtype=float)
        for _, y in train_data:
            # 获取样本标签值（支持张量或数值格式）
            label = int(y.item()) if hasattr(y, "item") else int(y)
            if label < 0 or label >= self.num_classes:
                continue  # 跳过异常标签
            self.label_count[label] += 1
        # 差分隐私噪声添加：拉普拉斯机制
        self.dp_epsilon = getattr(args, "dp_epsilon", 0.0)
        if self.dp_epsilon and self.dp_epsilon > 0:
            scale = 1.0 / self.dp_epsilon
            noise = np.random.laplace(loc=0.0, scale=scale, size=self.label_count.shape)
            self.label_count += noise
            # 裁剪使计数不为负
            self.label_count = np.clip(self.label_count, a_min=0.0, a_max=None)
        # 计算本地标签分布熵 H(k)，供服务器筛选和计算方差惩罚
        total = float(self.label_count.sum())
        if total > 0:
            p = self.label_count / total
            p_nonzero = p[p > 0]
            # 计算熵（以2为底）
            self.local_entropy = float(-np.sum(p_nonzero * np.log2(p_nonzero)))
        else:
            self.local_entropy = 0.0
