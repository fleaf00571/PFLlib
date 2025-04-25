import os
import json
import copy
import random
import numpy as np
from collections import deque, defaultdict

from flcore.servers.serveravg import FedAvg
from flcore.clients.clientfedentplus import clientFedEntPlus
from utils.data_utils import read_client_data


class FedEntPlusDynamic(FedAvg):
    def __init__(self, args, times):
        """
        动态参数配置版FedEntPlus
        
        Args:
            args: 包含配置文件路径等参数
            times: 随机种子编号
        """
        super().__init__(args, times)
        
        # 重建客户端列表，使用FedEntPlus专用客户端类
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
        
        # 计算平均数据量用于数据加权
        self.avg_samples = total_train_samples / self.num_clients if self.num_clients > 0 else 0.0
        
        # 预先存储每个客户端的标签计数和熵值，方便快速访问
        self.label_counts = [c.label_count.astype(float) for c in self.clients]
        self.local_entropies = [c.local_entropy for c in self.clients]
        
        # 策略和阶段跟踪
        self.current_stage_idx = 0    # 当前阶段索引
        self.current_strategy_idx = 0  # 当前策略索引
        self.strategy_round_count = 0  # 当前策略已执行轮次
        
        # 当前生效的参数
        self.current_params = {
            "H_min": getattr(args, "H_min", 0.0),
            "alpha_vp": getattr(args, "alpha_vp", 0.0),
            "beta": getattr(args, "beta", 0.0),
            "buffer_size": getattr(args, "buffer_size", 0)
        }
        
        # 每个策略的独立缓冲区 {(stage_idx, strategy_idx): deque()}
        self.strategy_buffers = defaultdict(lambda: deque())
        self.current_buffer = None  # 指向当前活跃的缓冲区
        
        # 解析配置文件
        config_path = getattr(args, "dynamic_config_path", "fedentplus_dynamic_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"动态配置文件未找到: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 验证和预处理配置
        self.stage_info = config.get("stage_info", [])
        if not self.stage_info:
            raise ValueError("配置中未找到'stage_info'")
            
        # 确保阶段按end_round排序
        self.stage_info.sort(key=lambda stage: stage["end_round"])
        
        # 日志选项
        log_config = config.get("logging", {})
        self.log_strategy_change = log_config.get("log_strategy_change", False)
        self.log_parameter_values = log_config.get("log_parameter_values", False)
        
        # 初始化阶段和策略
        self.initialize_stages_and_strategies()
    
    def initialize_stages_and_strategies(self):
        """初始化阶段和策略，设置初始参数和缓冲区"""
        if not self.stage_info:
            return
            
        # 为每个阶段的每个策略创建固定大小的缓冲区
        for stage_idx, stage in enumerate(self.stage_info):
            strategies = stage.get("strategies", [])
            for strategy_idx, strategy in enumerate(strategies):
                buffer_size = strategy["parameters"]["buffer_size"]
                # 创建指定大小的缓冲区
                self.strategy_buffers[(stage_idx, strategy_idx)] = deque(maxlen=buffer_size)
        
        # 设置初始策略的参数
        self.update_current_strategy(0)  # 从第0轮开始
    
    def update_current_strategy(self, round_idx):
        """
        根据当前轮次更新活跃的阶段和策略
        
        Args:
            round_idx: 当前训练轮次
        """
        # 找到当前应处于的阶段
        new_stage_idx = self.current_stage_idx
        for idx, stage in enumerate(self.stage_info):
            if round_idx <= stage["end_round"]:
                new_stage_idx = idx
                break
            # 如果超过所有已定义阶段的end_round，使用最后一个阶段
            if idx == len(self.stage_info) - 1:
                new_stage_idx = idx
        
        # 处理阶段转换
        if new_stage_idx != self.current_stage_idx:
            if self.log_strategy_change:
                print(f"[轮次 {round_idx}] 阶段变更: {self.current_stage_idx} -> {new_stage_idx}")
            
            # 阶段改变，重置策略索引和计数
            self.current_stage_idx = new_stage_idx
            self.current_strategy_idx = 0
            self.strategy_round_count = 0
        else:
            # 在同一阶段内，检查是否需要切换策略
            current_stage = self.stage_info[self.current_stage_idx]
            strategies = current_stage.get("strategies", [])
            
            if strategies:  # 确保有策略定义
                current_strategy = strategies[self.current_strategy_idx]
                strategy_duration = current_strategy.get("duration", 1)
                
                # 如果当前策略已完成其持续时间，切换到下一个
                if self.strategy_round_count >= strategy_duration:
                    # 计算下一个策略索引（循环使用策略列表）
                    self.current_strategy_idx = (self.current_strategy_idx + 1) % len(strategies)
                    self.strategy_round_count = 0
                    
                    if self.log_strategy_change:
                        strategy_id = strategies[self.current_strategy_idx].get("id", f"策略{self.current_strategy_idx}")
                        print(f"[轮次 {round_idx}] 切换到策略: {strategy_id}")
        
        # 更新当前参数和缓冲区
        self.update_current_parameters()
        self.update_current_buffer()
        
        # 增加当前策略的轮次计数
        self.strategy_round_count += 1
    
    def update_current_parameters(self):
        """更新当前活跃策略的参数"""
        if self.current_stage_idx >= len(self.stage_info):
            return
            
        current_stage = self.stage_info[self.current_stage_idx]
        strategies = current_stage.get("strategies", [])
        
        if not strategies:
            return
            
        # 获取当前策略的参数
        strategy = strategies[self.current_strategy_idx]
        params = strategy.get("parameters", {})
        
        # 更新当前参数
        self.current_params = {
            "H_min": params.get("H_min", 0.0),
            "alpha_vp": params.get("alpha_vp", 0.0),
            "beta": params.get("beta", 0.0),
            "buffer_size": params.get("buffer_size", 0)
        }
        
        # 更新args中的参数，以便在select_clients中使用
        self.args.H_min = self.current_params["H_min"]
        self.args.alpha_vp = self.current_params["alpha_vp"]
        self.args.beta = self.current_params["beta"]
        self.args.buffer_size = self.current_params["buffer_size"]
        
        if self.log_parameter_values:
            print(f"[参数更新] H_min={self.args.H_min}, alpha_vp={self.args.alpha_vp}, "
                  f"beta={self.args.beta}, buffer_size={self.args.buffer_size}")
    
    def update_current_buffer(self):
        """更新当前活跃的客户端选择缓冲区"""
        buffer_key = (self.current_stage_idx, self.current_strategy_idx)
        self.current_buffer = self.strategy_buffers[buffer_key]
    
    def select_clients(self):
        """
        根据当前策略参数选择客户端
        
        Returns:
            list: 选定的客户端列表
        """
        # 获取当前轮次（如果在train方法中设置）
        round_idx = getattr(self, "current_round", 0)
        
        # 更新当前策略和参数
        self.update_current_strategy(round_idx)
        
        # 调用修改版的FedEntPlus客户端选择逻辑
        return self._select_clients_dynamic()
    
    def _select_clients_dynamic(self):
        """基于当前参数执行FedEntPlus的客户端选择逻辑"""
        # 确定每轮选取客户端数量
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients+1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        M = self.current_num_join_clients
        
        # 初始候选池：排除当前缓冲区中的客户端
        if self.args.buffer_size > 0 and self.current_buffer:
            candidates = [c for c in self.clients if c.id not in self.current_buffer]
        else:
            candidates = self.clients.copy()
        
        # 熵下限过滤
        H_min = self.args.H_min
        if H_min is not None:
            candidates = [c for c in candidates if self.local_entropies[c.id] >= H_min]
            
        # 若过滤后候选池为空，则放宽限制
        if len(candidates) == 0:
            if H_min is not None:
                candidates = [c for c in self.clients if c.id not in self.current_buffer] \
                             if self.args.buffer_size > 0 and self.current_buffer else self.clients.copy()
            if len(candidates) == 0:
                candidates = self.clients.copy()
        
        # 初始化选择结果
        selected_clients = []
        selected_ids = set()
        
        # 随机选择首个客户端
        if len(candidates) > 0:
            first_client = random.choice(candidates)
        else:
            first_client = random.choice(self.clients)
        
        selected_clients.append(first_client)
        selected_ids.add(first_client.id)
        combined_count = self.label_counts[first_client.id].copy()
        
        if first_client in candidates:
            candidates.remove(first_client)
        
        # 迭代选择剩余客户端
        alpha_vp = self.args.alpha_vp
        beta = self.args.beta
        
        for _ in range(M - 1):
            # 如果候选池已空，尝试放宽限制
            if len(candidates) == 0:
                if self.args.buffer_size > 0:
                    candidates = [c for c in self.clients if c.id not in selected_ids]
                if len(candidates) == 0:
                    break
            
            best_score = -float('inf')
            best_client = None
            best_new_count = None
            
            # 确定未覆盖的标签集合
            total_samples = combined_count.sum()
            covered_labels = set(np.where(combined_count > 0)[0]) if total_samples > 0 else set()
            all_labels = set(range(self.num_classes))
            uncovered_labels = all_labels - covered_labels
            
            # 优先考虑能提供未覆盖标签的客户端
            if uncovered_labels:
                candidates_with_new_label = []
                for c in candidates:
                    client_labels = np.where(self.label_counts[c.id] > 0)[0]
                    if len(uncovered_labels.intersection(client_labels)) > 0:
                        candidates_with_new_label.append(c)
                candidates_subset = candidates_with_new_label if candidates_with_new_label else candidates
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
                
                # 计算熵集合方差
                new_entropy_list = [self.local_entropies[i] for i in selected_ids] + [self.local_entropies[c.id]]
                entropy_var = float(np.var(new_entropy_list))
                
                # 计算数据量调节因子
                data_factor = c.train_samples / self.avg_samples if self.avg_samples > 0 else 1.0
                
                # 计算综合评分
                score = H_new - alpha_vp * entropy_var + beta * data_factor
                
                if score > best_score:
                    best_score = score
                    best_client = c
                    best_new_count = new_count
            
            # 选择得分最高的客户端
            if best_client is None:
                break
                
            selected_clients.append(best_client)
            selected_ids.add(best_client.id)
            combined_count = best_new_count
            candidates.remove(best_client)
        
        # 更新当前缓冲区
        if self.args.buffer_size > 0 and self.current_buffer is not None:
            for c in selected_clients:
                self.current_buffer.append(c.id)
        
        return selected_clients
    
    def train(self):
        """执行训练过程，每轮更新当前轮次计数并选择客户端"""
        for i in range(self.global_rounds + 1):
            self.current_round = i  # 设置当前轮次，供select_clients使用
            
            # 选择客户端
            self.selected_clients = self.select_clients()
            self.send_models()
            
            # 定期评估
            if i % self.eval_gap == 0:
                print(f"\n--- 轮次 {i}/{self.global_rounds} ---")
                self.evaluate()
            
            # 训练客户端
            for client in self.selected_clients:
                client.train()
            
            # 接收和聚合模型
            self.receive_models()
            self.aggregate_parameters()
            
            # 检查是否可以提前终止
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        
        print("\n========== 训练完成 ==========")
        print("最佳测试准确率:", max(self.rs_test_acc) if self.rs_test_acc else None)
        self.save_results()
        self.save_global_model()