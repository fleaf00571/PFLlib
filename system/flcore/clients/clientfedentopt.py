import time
import numpy as np
import torch

from flcore.clients.clientavg import clientAVG
from utils.data_utils import read_client_data  # 修改为正确路径

class ClientEntOptDP(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        # 给 train_slow / send_slow 默认值
        kwargs.setdefault('train_slow', False)
        kwargs.setdefault('send_slow', False)

        # 这里的 train_samples / test_samples 是 int
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # 如果你还想访问实际训练数据，就从 kwargs 拿
        real_train_data = kwargs.get('real_train_data', None)
        # real_test_data  = kwargs.get('real_test_data', None)

        # 统计标签分布
        self.label_count = np.zeros(args.num_classes, dtype=float)
        if real_train_data is not None:
            for _, y in real_train_data:
                label = int(y.item()) if hasattr(y, "item") else int(y)
                self.label_count[label] += 1

        # 拉普拉斯加噪
        self.dp_epsilon = getattr(args, "dp_epsilon", 0.0)
        if self.dp_epsilon > 0:
            scale = 1.0 / self.dp_epsilon
            noise = np.random.laplace(loc=0.0, scale=scale, size=self.label_count.shape)
            self.label_count += noise
            self.label_count = np.clip(self.label_count, a_min=0.0, a_max=None)
