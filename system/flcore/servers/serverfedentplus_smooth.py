# serverfedentplus_smooth.py

import os
import json
import random
import copy
import numpy as np
from collections import deque

from flcore.servers.serveravg import FedAvg  # 用于复用基本训练流程
from flcore.clients.clientfedentplus import clientFedEntPlus  # 客户端类
from utils.data_utils import read_client_data

# 固定平滑过渡轮次
TRANSITION_ROUNDS = 10

class FedEntPlusSmooth(FedAvg):
    def __init__(self, args, times):
        """
        args: 其中可以包含 dynamic_config_path 等参数
        times: 随机种子编号
        """
        super().__init__(args, times)

        # 重新构建客户端列表：使用clientFedEntPlus，以便支持本地熵计算
        self.clients = []
        total_train_samples = 0
        for cid in range(self.num_clients):
            train_data = read_client_data(self.dataset, cid, is_train=True)
            test_data = read_client_data(self.dataset, cid, is_train=False)
            train_samples = len(train_data)
            test_samples = len(test_data)
            total_train_samples += train_samples

            c = clientFedEntPlus(
                args, cid,
                train_samples, test_samples,
                real_train_data=train_data,
                real_test_data=test_data
            )
            self.clients.append(c)

        # 平均数据量，用于数据奖励系数的归一化
        self.avg_samples = (
            total_train_samples / self.num_clients
            if self.num_clients > 0 else 0.0
        )

        # 初始化 FIFO 缓冲区
        self.buffer_size = getattr(args, "buffer_size", 0)
        self.selection_buffer = deque(maxlen=self.buffer_size)

        # 预存每个客户端的标签计数/熵
        self.label_counts = [c.label_count.astype(float) for c in self.clients]
        self.local_entropies = [c.local_entropy for c in self.clients]

        # =========== 读取 JSON 配置并进行预处理 ============
        # 默认配置文件路径，可在args中通过 dynamic_config_path 修改
        config_path = getattr(args, "dynamic_config_path", "fedentplus_smooth_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Dynamic config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)
        stage_info = config.get("stage_info", [])
        if not stage_info:
            raise ValueError("No 'stage_info' found in config.")

        # 将 stage_info 按 end_fraction 升序排序
        stage_info = sorted(stage_info, key=lambda s: s["end_fraction"])

        self.total_rounds = self.global_rounds  # 总轮数
        # 把 stage_info 转换成 [ (end_round, param_dict), ... ] 形式
        self.stages = []
        for st in stage_info:
            # 若使用 fraction => end_round = int(fraction * total_rounds)
            # 若要用绝对轮次，直接把 st["end_round"] 作为 end_round 即可
            fraction = st["end_fraction"]
            end_r = int(round(fraction * self.total_rounds))
            param_dict = {
                "H_min": st["H_min"],
                "alpha_vp": st["alpha_vp"],
                "beta": st["beta"],
                "buffer_size": st["buffer_size"],
            }
            self.stages.append((end_r, param_dict))

        # stage 参数（上一阶段末的固定值）
        self.curr_params = None  # 当前实际使用的参数
        self.next_params = None  # 下阶段目标参数
        self.stage_idx = 0       # 正在进行哪个阶段
        self.stage_start_round = 0  # 当前阶段的起始轮次(用于平滑过渡)
        self.last_stage_params = None   # 记录上阶段末的参数
        self.load_initial_params()

    def load_initial_params(self):
        """
        读取第一阶段的目标参数，先初始化成 curr_params = next_params = 第一个阶段目标。
        若训练从 round=0 开始，就默认进入 stage_idx=0。
        """
        # 第一阶段若 stage_info[0] 不是从 round=0 开始，也当作0开始
        if len(self.stages) == 0:
            # 若无阶段信息则使用默认
            self.curr_params = {
                "H_min": getattr(self.args, "H_min", 0.0),
                "alpha_vp": getattr(self.args, "alpha_vp", 0.0),
                "beta": getattr(self.args, "beta", 0.0),
                "buffer_size": getattr(self.args, "buffer_size", 0),
            }
            self.last_stage_params = copy.deepcopy(self.curr_params)
            self.next_params = copy.deepcopy(self.curr_params)
            return

        # 取阶段 0 的参数作为初始值
        self.stage_idx = 0
        self.next_params = copy.deepcopy(self.stages[0][1])
        self.curr_params = copy.deepcopy(self.stages[0][1])
        self.last_stage_params = copy.deepcopy(self.stages[0][1])

    def update_stage_if_needed(self, round_idx):
        """
        判断当前轮次是否需要进入下一个阶段；若进入下阶段则触发： 
        1）重置 stage_idx 
        2）记下 last_stage_params = 上阶段末参数 
        3）更新 next_params = 下阶段目标参数 
        4）更新 stage_start_round = round_idx
        5）buffer_size 直接生效
        """
        # 如果已经在最后一个阶段，就不再更新
        if self.stage_idx >= len(self.stages):
            return

        end_r = self.stages[self.stage_idx][0]
        # 若当前轮次超出当前阶段 end_r，说明需要进入下一个阶段
        if round_idx > end_r:
            self.stage_idx += 1
            # 若越界，则保持最后
            if self.stage_idx >= len(self.stages):
                self.stage_idx = len(self.stages) - 1
                return

            # 正式进入下一个阶段
            self.last_stage_params = copy.deepcopy(self.curr_params)
            self.next_params = copy.deepcopy(self.stages[self.stage_idx][1])
            self.stage_start_round = round_idx

            # buffer_size 改变 => 立即生效
            old_buf = self.buffer_size
            new_buf = self.next_params["buffer_size"]
            if new_buf != old_buf:
                self.adjust_buffer_size(old_buf, new_buf)
            self.buffer_size = new_buf

    def adjust_buffer_size(self, old_buf, new_buf):
        """
        buffer_size 立即调整：
        - 若 new_buf > old_buf，则增大 selection_buffer 的 maxlen
        - 若 new_buf < old_buf，则减小，同时剔除队列前面的多余客户端
        """
        # 先把已有client id记录
        old_ids = list(self.selection_buffer)
        self.selection_buffer = deque(maxlen=new_buf)

        if new_buf >= old_buf:
            # 扩大容量 => 依次插回旧记录(都能放进去)
            for cid in old_ids:
                self.selection_buffer.append(cid)
        else:
            # 减小容量 => 只保留旧队列最靠后的 new_buf 个
            # 这里“最前面”指最早加入的，需要剔除
            remain_ids = old_ids[-new_buf:]  # 取后 new_buf 个
            for cid in remain_ids:
                self.selection_buffer.append(cid)

    def compute_smooth_params(self, round_idx):
        """
        在每轮根据上一阶段末参数(last_stage_params)和下一阶段目标(next_params)，
        在前 TRANSITION_ROUNDS 内线性插值，之后保持 next_params 不变。
        如果某个参数 old == new，则无需插值。
        """
        # 若当前阶段只有一个参数区间，或已经超过最后阶段，也直接返回 curr_params
        if self.stage_idx >= len(self.stages):
            return self.curr_params

        # 当前阶段的 end_r
        end_r = self.stages[self.stage_idx][0]

        # 判断是否仍在平滑过渡区间
        rounds_since_stage_start = round_idx - self.stage_start_round
        # 当 rounds_since_stage_start <= TRANSITION_ROUNDS 时，我们插值
        # 超过则直接用 next_params
        if rounds_since_stage_start < 0:
            rounds_since_stage_start = 0  # 可能出现round=0情况

        new_params = {}
        for k in ["H_min", "alpha_vp", "beta", "buffer_size"]:
            old_val = self.last_stage_params[k]
            target_val = self.next_params[k]
            if abs(target_val - old_val) < 1e-12:
                # 无变化
                new_params[k] = target_val
            else:
                if rounds_since_stage_start <= TRANSITION_ROUNDS:
                    ratio = rounds_since_stage_start / float(TRANSITION_ROUNDS)
                    val = old_val + ratio * (target_val - old_val)
                    # buffer_size 不平滑，这里只要保留 old_val 或 target_val？
                    # 根据需求：buffer_size 不参与平滑 => 直接用 target_val
                    if k == "buffer_size":
                        new_params[k] = target_val
                    else:
                        new_params[k] = val
                else:
                    # 平滑期已结束
                    new_params[k] = target_val
        return new_params

    def select_clients(self):
        """
        在选客户端前更新阶段、计算平滑参数并应用，然后执行 FedEntPlus 的选客户端逻辑。
        """
        # 若在 train() 中，会将 self.current_round 传给这里
        round_idx = getattr(self, "current_round", 0)

        # 1) 是否需要进入下一个阶段
        self.update_stage_if_needed(round_idx)

        # 2) 计算当前平滑后的参数
        smoothed = self.compute_smooth_params(round_idx)
        self.curr_params = smoothed

        # 3) 更新到 self.args 中，以让父类逻辑使用
        self.args.H_min = smoothed["H_min"]
        self.args.alpha_vp = smoothed["alpha_vp"]
        self.args.beta = smoothed["beta"]
        # buffer_size 已在进入阶段时立即调整，因此此处无需修改 selection_buffer

        # 4) 调用原本的 FedEntPlus 选择逻辑（复制原 FedEntPlus 里 select_clients 代码即可）
        return self._select_clients_fedentplus()

    def _select_clients_fedentplus(self):
        """
        原 FedEntPlus.select_clients 的核心逻辑：包含熵阈值过滤、缓冲区排除等。
        """
        # 确定每轮选客户端数量
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients+1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        M = self.current_num_join_clients

        # 初始候选池：排除缓冲区中的客户端
        if self.buffer_size > 0:
            candidates = [c for c in self.clients if c.id not in self.selection_buffer]
        else:
            candidates = self.clients.copy()

        # 熵下限过滤
        H_min_val = getattr(self.args, "H_min", None)
        if H_min_val is not None:
            candidates = [c for c in candidates if self.local_entropies[c.id] >= H_min_val]

        # 若过滤后为空 => 放宽限制
        if len(candidates) == 0:
            if H_min_val is not None:
                # 撤销熵过滤
                candidates = [c for c in self.clients if c.id not in self.selection_buffer] \
                    if self.buffer_size > 0 else self.clients.copy()
            if len(candidates) == 0:
                candidates = self.clients.copy()

        # 开始按照多样性优先策略进行选择
        selected_clients = []
        selected_ids = set()

        # 1) 随机选首个
        if len(candidates) > 0:
            first_client = random.choice(candidates)
        else:
            first_client = random.choice(self.clients)
        selected_clients.append(first_client)
        selected_ids.add(first_client.id)
        combined_count = self.label_counts[first_client.id].copy()
        candidates.remove(first_client) if first_client in candidates else None

        # 2) 迭代选择剩余 M-1 个
        alpha_vp = getattr(self.args, "alpha_vp", 0.0)
        beta = getattr(self.args, "beta", 0.0)

        for _ in range(M - 1):
            if len(candidates) == 0:
                # 尝试放宽缓冲限制
                if self.buffer_size > 0:
                    candidates = [c for c in self.clients if (c.id not in selected_ids)]
                if len(candidates) == 0:
                    break

            best_score = -float('inf')
            best_client = None
            best_new_count = None

            total_samples = combined_count.sum()
            covered_labels = set(np.where(combined_count > 0)[0]) if total_samples > 0 else set()
            all_labels = set(range(self.num_classes))
            uncovered_labels = all_labels - covered_labels

            # 若有未覆盖标签 => 限制候选仅包含能提供新标签的客户端
            if uncovered_labels:
                candidates_with_new_label = []
                for c in candidates:
                    client_labels = np.where(self.label_counts[c.id] > 0)[0]
                    if len(uncovered_labels.intersection(client_labels)) > 0:
                        candidates_with_new_label.append(c)
                candidates_subset = candidates_with_new_label if len(candidates_with_new_label) > 0 else candidates
            else:
                candidates_subset = candidates

            # 遍历候选，计算综合评分
            for c in candidates_subset:
                new_count = combined_count + self.label_counts[c.id]
                tot = new_count.sum()
                if tot <= 0:
                    continue
                p = new_count / tot
                p_nonzero = p[p > 0]
                H_new = -np.sum(p_nonzero * np.log2(p_nonzero))

                # 方差惩罚
                new_entropy_list = [self.local_entropies[i] for i in selected_ids] + [self.local_entropies[c.id]]
                entropy_var = float(np.var(new_entropy_list))

                # 数据量加成
                data_factor = (c.train_samples / self.avg_samples) if self.avg_samples > 0 else 1.0

                # 评分公式
                score = H_new - alpha_vp * entropy_var + beta * data_factor
                if score > best_score:
                    best_score = score
                    best_client = c
                    best_new_count = new_count

            if best_client is None:
                break

            selected_clients.append(best_client)
            selected_ids.add(best_client.id)
            combined_count = best_new_count
            candidates.remove(best_client)

        # 更新缓冲区
        if self.buffer_size > 0:
            for sc in selected_clients:
                self.selection_buffer.append(sc.id)

        return selected_clients

    def train(self):
        """
        训练流程：每轮更新 self.current_round => 动态参数更新 => 选客户端 => 下发 => 客户端训练 => 聚合
        """
        for i in range(self.global_rounds + 1):
            self.current_round = i

            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n--- Round {i}/{self.global_rounds} ---")
                self.evaluate()

            # 训练客户端
            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            # 可选：若需要提早终止
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\n========== Training Complete ==========")
        print("Best Test Accuracy:", max(self.rs_test_acc) if self.rs_test_acc else None)
        self.save_results()
        self.save_global_model()
