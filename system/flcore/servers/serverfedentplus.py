import random
import numpy as np
from collections import deque
from flcore.servers.serveravg import FedAvg  # 继承FedAvg以复用基础训练流程
from flcore.clients.clientfedentplus import clientFedEntPlus  # 导入对应的客户端类
from utils.data_utils import read_client_data


class FedEntPlus(FedAvg):
    def __init__(self, args, times):
        # 调用 FedAvg 构造函数（其内部会设置基本属性并创建初始 clients 列表）
        super().__init__(args, times)
        # -------- 重建客户端列表，使用 FedEntPlus 专用客户端类 --------
        # 利用数据工具读取每个客户端的本地数据，并初始化对应的 clientFedEntPlus 对象
        self.clients = []
        total_train_samples = 0
        for cid in range(self.num_clients):
            # 获取客户端 cid 的原始训练和测试数据（列表形式）
            train_data = read_client_data(self.dataset, cid, is_train=True)
            test_data = read_client_data(self.dataset, cid, is_train=False)
            train_samples = len(train_data)
            test_samples = len(test_data)
            total_train_samples += train_samples
            # 创建 FedEntPlus 客户端对象，传入真实数据列表以便统计标签分布
            client = clientFedEntPlus(
                args, cid,
                train_samples, 
                test_samples,
                real_train_data=train_data,
                real_test_data=test_data
            )
            self.clients.append(client)
        # 计算平均数据量用于数据加权（f(|D|) 可取为客户端数据量与平均值之比）
        self.avg_samples = total_train_samples / self.num_clients if self.num_clients > 0 else 0.0
        # 初始化 FIFO 缓冲队列 B，长度 Q 默认为 args.buffer_size（若未设置则为0表示不使用缓冲）
        self.buffer_size = getattr(args, "buffer_size", 0)
        self.selection_buffer = deque(maxlen=self.buffer_size)  # 超过长度时自动弹出旧元素
        # 预先存储每个客户端的标签计数和熵值，方便快速访问
        self.label_counts = [client.label_count.astype(float) for client in self.clients]
        self.local_entropies = [client.local_entropy for client in self.clients]

    def select_clients(self):
        """选择一批客户端组成当前轮参与集合，优化其全局标签分布熵并应用熵/数据多样性约束。"""
        # 确定每轮选取客户端数量 M（支持随机参与率）
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        M = self.current_num_join_clients

        # 初始候选池 A：排除缓冲区B中的客户端
        if self.buffer_size > 0:
            candidates = [c for c in self.clients if c.id not in self.selection_buffer]
        else:
            candidates = self.clients.copy()
        # 熵下限过滤：移除本地熵低于阈值 H_min 的客户端
        H_min = getattr(self.args, "H_min", None)
        if H_min is not None:
            candidates = [c for c in candidates if self.local_entropies[c.id] >= H_min]
        # 若过滤后候选池为空，则放宽限制（首先解除熵阈值限制，其次如仍为空则包括缓冲中的客户端）
        if len(candidates) == 0:
            # 如果由于熵阈值导致为空，则撤销熵过滤
            if H_min is not None:
                candidates = [c for c in (self.clients) if c.id not in self.selection_buffer] \
                             if self.buffer_size > 0 else self.clients.copy()
            # 如果候选仍为空且缓冲机制存在，则进一步解除缓冲限制
            if len(candidates) == 0:
                candidates = self.clients.copy()
        # 初始化选择结果集合 S
        selected_clients = []
        selected_ids = set()

        # 1. 随机选择首个客户端（增加探索多样性）&#8203;:contentReference[oaicite:9]{index=9}
        if len(candidates) > 0:
            first_client = random.choice(candidates)
        else:
            first_client = random.choice(self.clients)  # 极端情况下（候选为空）从所有中随机选
        selected_clients.append(first_client)
        selected_ids.add(first_client.id)
        # 更新累计的标签计数向量 L 为所选客户端的标签计数
        combined_count = self.label_counts[first_client.id].copy()
        # 从候选池中移除已选客户端
        if first_client in candidates:
            candidates.remove(first_client)

        # 2. 迭代选择剩余客户端&#8203;:contentReference[oaicite:10]{index=10}
        alpha_vp = getattr(self.args, "alpha_vp", 0.0)  # 熵方差惩罚系数 α，默认0表示不惩罚
        beta = getattr(self.args, "beta", 0.0)    # 数据量加权系数 β，默认0表示无加权
        # 逐个选择剩余的 M-1 个客户端
        for _ in range(M - 1):
            # 如果候选池已空，尝试放宽限制以继续选择
            if len(candidates) == 0:
                # 若缓冲限制尚有作用，解除之（允许之前轮次客户端重新参与）
                if self.buffer_size > 0:
                    # 引入此前缓冲的客户端作为候选，但排除已选过的和当前正在选的
                    candidates = [c for c in self.clients if c.id not in selected_ids]
                # 如果应用了熵阈值且此时候选依然不足，则进一步放宽熵限制
                if H_min is not None and len(candidates) == 0:
                    candidates = [c for c in self.clients if c.id not in selected_ids]
            if len(candidates) == 0:
                break  # 仍无候选可选，跳出循环（可能无法满足选满M个的情况）

            best_score = -float('inf')
            best_client = None
            best_new_count = None

            # 确定未覆盖的标签集合 U，用于类别覆盖优先&#8203;:contentReference[oaicite:11]{index=11}
            total_samples = combined_count.sum()
            covered_labels = set(np.where(combined_count > 0)[0]) if total_samples > 0 else set()
            all_labels = set(range(self.num_classes))
            uncovered_labels = all_labels - covered_labels
            # 如果存在尚未覆盖的标签，则限制候选池仅包括能提供这些标签的客户端&#8203;:contentReference[oaicite:12]{index=12}
            if uncovered_labels:
                # 在当前候选列表中筛选包含未覆盖标签的客户端
                candidates_with_new_label = []
                for c in candidates:
                    # 检查客户端c是否拥有任一未覆盖的标签
                    client_labels = np.where(self.label_counts[c.id] > 0)[0]
                    if len(uncovered_labels.intersection(client_labels)) > 0:
                        candidates_with_new_label.append(c)
                if len(candidates_with_new_label) > 0:
                    # 存在可以引入新标签的客户端，则仅在这些客户端中选择
                    candidates_subset = candidates_with_new_label
                else:
                    candidates_subset = candidates
            else:
                candidates_subset = candidates

            # 遍历当前候选，计算加入后的评分 score(j)&#8203;:contentReference[oaicite:13]{index=13}
            for c in candidates_subset:
                # 计算将客户端 c 加入当前集合后的合并标签分布熵 H_new
                new_count = combined_count + self.label_counts[c.id]
                total = new_count.sum()
                if total <= 0:
                    continue  # 跳过无数据的客户端
                p = new_count / total
                # 计算全局分布熵 H_new （以2为底的熵）
                p_nonzero = p[p > 0]
                H_new = -np.sum(p_nonzero * np.log2(p_nonzero))
                # 计算加入该客户端后的熵集合方差 Var(H_{S+j})&#8203;:contentReference[oaicite:14]{index=14}
                new_entropy_list = [self.local_entropies[i] for i in selected_ids] + [self.local_entropies[c.id]]
                entropy_var = float(np.var(new_entropy_list))  # 转为普通 float
                # 计算数据量调节因子 f(|D_c|) = |D_c| / 平均数据量，用于加权&#8203;:contentReference[oaicite:15]{index=15}
                data_factor = c.train_samples / self.avg_samples if self.avg_samples > 0 else 1.0
                # 计算综合评分：score = H_new - α * Var(H) + β * data_factor
                score = H_new - alpha_vp * entropy_var + beta * data_factor
                if score > best_score:
                    best_score = score
                    best_client = c
                    best_new_count = new_count

            # 从候选池中选择得分最高的客户端
            if best_client is None:
                break  # 无合适客户端（可能所有候选熵增益为0），结束选择
            selected_clients.append(best_client)
            selected_ids.add(best_client.id)
            # 更新累计分布 L 和候选池
            combined_count = best_new_count
            candidates.remove(best_client)
        # 3. 更新缓冲区B：将本轮选中的客户端加入缓冲&#8203;:contentReference[oaicite:16]{index=16}
        if self.buffer_size > 0:
            for client in selected_clients:
                self.selection_buffer.append(client.id)
        return selected_clients
    
    

