import random
import numpy as np
from collections import deque
from flcore.servers.serveravg import FedAvg
from utils.data_utils import read_client_data
# 修改为从 clientfedentopt 导入 ClientEntOptDP
from flcore.clients.clientfedentopt import ClientEntOptDP

class FedEntOpt(FedAvg):
    def __init__(self, args, times):
        # 先调用父类FedAvg的构造（该构造会设置 self.dataset, self.num_clients 等）
        super().__init__(args, times)

        # ========== 替换原有客户端创建 ==========
        # 使用 ClientEntOptDP 来创建客户端对象
        from utils.data_utils import read_client_data  # 确保导入正确
        self.clients = []
        for cid in range(self.num_clients):
            # 先获取实际数据列表
            raw_train_data = read_client_data(self.dataset, cid, is_train=True)
            raw_test_data = read_client_data(self.dataset, cid, is_train=False)

            # 这里把 整型长度 赋给 train_samples / test_samples
            train_samples = len(raw_train_data)
            test_samples = len(raw_test_data)

            # 同时，用别的字段（如 real_train_data / real_test_data）存原始列表
            c = ClientEntOptDP(
                args, cid,
                train_samples,  # 纯粹是 int
                test_samples,
                real_train_data=raw_train_data,  # 额外传真正数据
                real_test_data=raw_test_data
            )
            self.clients.append(c)

        # FIFO buffer：根据 args.buffer_size 设置
        self.buffer_size = getattr(args, "buffer_size", 0)
        self.selection_buffer = deque(maxlen=self.buffer_size) if self.buffer_size > 0 else deque()

        # 不再读取原始数据统计；直接从每个客户端的 label_count 获取
        self.label_counts = []
        for c in self.clients:
            self.label_counts.append(c.label_count.astype(float))

    def select_clients(self):
        """Select a subset of clients maximizing the entropy of their aggregated label distribution."""
        # 与原有逻辑保持一致
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        M = self.current_num_join_clients

        if self.buffer_size > 0:
            eligible_pool = [c for c in self.clients if c.id not in self.selection_buffer]
        else:
            eligible_pool = self.clients.copy()
        if len(eligible_pool) == 0:
            eligible_pool = self.clients.copy()

        selected_clients = []
        selected_ids = set()

        # 1. Randomly select the first client
        first_client = random.choice(eligible_pool)
        selected_clients.append(first_client)
        selected_ids.add(first_client.id)
        combined_count = self.label_counts[first_client.id].copy()
        eligible_pool.remove(first_client)

        # 2. Iteratively select remaining clients based on entropy maximization
        for _ in range(M - 1):
            if len(eligible_pool) == 0:
                break
            best_entropy = -1.0
            best_client = None
            best_new_count = None
            for client in eligible_pool:
                new_count = combined_count + self.label_counts[client.id]
                total_samples = new_count.sum()
                if total_samples == 0:
                    continue
                p = new_count / total_samples
                entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
                if entropy > best_entropy:
                    best_entropy = entropy
                    best_client = client
                    best_new_count = new_count
            if best_client is None:
                break
            selected_clients.append(best_client)
            selected_ids.add(best_client.id)
            combined_count = best_new_count
            eligible_pool.remove(best_client)
            if len(eligible_pool) == 0 and len(selected_clients) < M:
                remaining_pool = [c for c in self.clients if c.id not in selected_ids]
                eligible_pool = remaining_pool.copy()

        if self.buffer_size > 0:
            for client in selected_clients:
                self.selection_buffer.append(client.id)

        return selected_clients
