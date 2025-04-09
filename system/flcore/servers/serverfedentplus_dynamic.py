import time
import json
import os
import numpy as np
from collections import deque
from flcore.servers.serverfedentplus import FedEntPlus  # 基于FedEntPlus机制
from utils.data_utils import read_client_data

class FedEntPlusDynamic(FedEntPlus):
    def __init__(self, args, times, log_filename=None):
        """
        FedEntPlusDynamic服务器初始化。
        如果启用动态参数调整，将按照预设阶段策略设置初始参数。
        如果启用日志记录，将打开日志文件并记录初始状态。
        """
        # 检查是否启用动态参数调整
        self.dynamic_param = getattr(args, "enable_dynamic_param", False)
        if self.dynamic_param:
            # 设置动态调整初始阶段的参数值
            args.H_min = 0.0       # qianqifangkuan熵阈值
            args.alpha_vp = 0.0    # 初期不惩罚熵差异
            args.beta = 0.0        # 初期不考虑数据量权重
            args.buffer_size = 30  # 初期较大缓冲区
        # 调用父类 FedEntPlus 初始化（这会建立客户端列表、计算平均数据量、初始化缓冲区等）
        super().__init__(args, times)
        # 保存参数和运行编号
        self.args = args
        self.times = times

        # 动态参数调整阈值（根据全局轮数计算中期和后期的轮次边界）
        self.half_round = int(0.3 * self.global_rounds)    # 中期开始轮次（约一半轮次时）
        self.three_quarter_round = int(0.8 * self.global_rounds)  # 后期开始轮次（约4/5轮次时）

        # 如果启用了日志记录，通过 --log_fedent，则初始化日志文件
        self.log_enabled = getattr(args, "log_fedent", False)
        if self.log_enabled:
            # 确定日志文件路径和名称（包含数据集、算法名称和运行编号）
            if log_filename is None:
                log_filename = f"{args.save_folder_name}/{args.dataset}_{args.algorithm}_run{times}_log.json"
            # 确保文件夹路径存在，如果不存在则创建
            log_dir = os.path.dirname(log_filename)
            if log_dir and not os.path.exists(log_dir):  # 检查目录是否存在
                os.makedirs(log_dir, exist_ok=True)  # 创建目录

            # 打开日志文件
            self.log_file = open(log_filename, "w", encoding="utf-8")
            
            # 记录训练开始时间和初始配置信息
            start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            training_start_info = {
                "start_time": start_time_str,
                "num_clients": self.num_clients,
                "global_rounds": self.global_rounds,
                "clients_per_round": self.num_join_clients,
                "random_join": self.random_join_ratio,
                "H_min": getattr(args, "H_min", None),
                "alpha_vp": getattr(args, "alpha_vp", 0.0),
                "beta": getattr(args, "beta", 0.0),
                "dp_epsilon": getattr(args, "dp_epsilon", 0.0),
                "buffer_size": getattr(args, "buffer_size", 0)
            }
            # 将初始配置写入日志文件为 JSON，并开始记录轮次数组
            init_json = json.dumps({"training_start": training_start_info}, ensure_ascii=False, indent=4)
            # 写入开头部分，准备写入轮次信息（保留 JSON 结构）
            self.log_file.write(init_json[:-1] + ',\n  "rounds": [\n')
            # 预计算所有客户端的原始标签分布（未加噪前）用于日志记录
            self.raw_label_counts = []
            for cid in range(self.num_clients):
                label_count = np.zeros(self.num_classes, dtype=int)
                train_data = read_client_data(self.dataset, cid, is_train=True)
                for _, y in train_data:
                    # 确保 y 转换为整型标签索引
                    label = int(y.item()) if hasattr(y, "item") else int(y)
                    if 0 <= label < self.num_classes:
                        label_count[label] += 1
                self.raw_label_counts.append(label_count.tolist())
            # 记录训练起始时间戳（秒）用于计算总耗时
            self.training_start_time = time.time()
        else:
            self.log_file = None
            self.raw_label_counts = None
            self.training_start_time = time.time()

    def train(self):
        """运行联邦训练过程（支持动态参数调整和可选日志记录）。"""
        # 遍历全局训练轮次
        for round_idx in range(1, self.global_rounds + 1):
            # ---------- 动态参数调整机制 ----------
            if self.dynamic_param:
                # 在中期阶段开始（总轮次30%处）调整参数：放宽熵阈值，提高数据量权重
                if round_idx == self.half_round:
                    self.args.H_min = 0.0    # 放宽本地熵下限阈值，允许熵更低的客户端参与
                    self.args.beta = 0.02    # 引入适度的数据量权重，利用大数据量客户端提升精度
                    # （缓冲区大小在中期默认保持不变，如有需要可在此调整）
                # 在后期阶段开始（总轮次80%处）调整参数：增加熵方差惩罚，减小缓冲区
                if round_idx == self.three_quarter_round:
                    self.args.H_min = 0.5  # 提高本地熵下限阈值，限制熵过低的客户端参与
                    self.args.alpha_vp = 0.2   # 提高熵方差惩罚系数，限制所选客户端熵差异过大
                    self.args.beta = 0.01  # 减小数据量权重，避免过度依赖大数据量客户端
                    new_buffer_size = 20       # 减小缓冲区长度
                    # 如果缓冲区大小发生变化，重新初始化缓冲队列以应用新的大小限制
                    if new_buffer_size != self.buffer_size:
                        old_buffer_elems = list(self.selection_buffer)  # 当前缓冲区内容
                        self.buffer_size = new_buffer_size
                        # 使用新长度创建新的 deque，并保留旧缓冲中最近的新_buffer_size 个元素
                        self.selection_buffer = deque(maxlen=self.buffer_size)
                        if len(old_buffer_elems) >= self.buffer_size:
                            to_keep = old_buffer_elems[-self.buffer_size:]  # 保留最新的部分
                        else:
                            to_keep = old_buffer_elems
                        for cid in to_keep:
                            self.selection_buffer.append(cid)
                        # 将 args.buffer_size 更新同步
                        setattr(self.args, "buffer_size", new_buffer_size)
            # ---------- 客户端选择与训练 ----------
            round_start_time = time.time()
            # 1. 选择本轮参与的客户端（使用 FedEntPlus 内置的选择算法，考虑当前参数和缓冲区）
            selected_clients = self.select_clients()
            self.selected_clients = selected_clients  # 保存选中客户端列表供后续步骤使用
            selected_ids = [c.id for c in selected_clients]

            # 2. 记录选中客户端的详细信息（用于日志）
            clients_info = {}
            if self.log_enabled:
                for client in selected_clients:
                    cid = client.id
                    clients_info[cid] = {
                        "raw_label_distribution": self.raw_label_counts[cid] if self.raw_label_counts is not None else None,
                        "dp_label_distribution": self.label_counts[cid].tolist(),
                        "local_entropy": float(self.local_entropies[cid]),
                        "train_samples": int(client.train_samples)
                    }
            # 3. 下发全局模型并执行各客户端本地训练
            self.send_models()               # 将当前全局模型发送给所有客户端
            for client in selected_clients:
                client.train()               # 每个选中客户端执行本地训练
            self.receive_models()            # 接收所有选中客户端的模型更新
            self.aggregate_parameters()      # 聚合更新全局模型参数

            # 4. 评估全局模型在测试集上的性能，并计算训练损失
            self.send_models()               # 确保所有客户端持有最新全局模型（用于测试评估）
            ids, num_samples, tot_correct, tot_auc = self.test_metrics()       # 收集测试指标
            _, num_train_samples, train_losses = self.train_metrics()          # 收集训练损失
            total_samples = sum(num_samples)
            test_accuracy = float(sum(tot_correct) / total_samples) if total_samples > 0 else 0.0
            test_auc = float(sum(tot_auc) / total_samples) if total_samples > 0 else 0.0
            avg_train_loss = float(sum(train_losses) / sum(num_train_samples)) if sum(num_train_samples) > 0 else 0.0
            performance_metrics = {
                "test_accuracy": test_accuracy,
                "test_auc": test_auc,
                "train_loss": avg_train_loss
            }

            # 5. 计算本轮合并后的全局标签分布及熵值
            combined_count = np.zeros(self.num_classes, dtype=float)
            for cid in selected_ids:
                combined_count += self.label_counts[cid]
            global_label_distribution = combined_count.tolist()
            total_count = combined_count.sum()
            if total_count > 0:
                p = combined_count / total_count
                p_nonzero = p[p > 0]
                global_entropy = float(-np.sum(p_nonzero * np.log2(p_nonzero)))
            else:
                global_entropy = 0.0

            # 6. 记录客户端选择过程的评分指标（按照选择顺序逐个记录）
            selection_metrics = []
            if self.log_enabled:
                current_combined_count = np.zeros(self.num_classes, dtype=float)
                current_selected_entropies = []
                alpha_vp = getattr(self.args, "alpha_vp", 0.0)
                beta = getattr(self.args, "beta", 0.0)
                for client in selected_clients:
                    cid = client.id
                    # 计算选中该客户端前的全局熵 H_old
                    prev_total = current_combined_count.sum()
                    if prev_total > 0:
                        prev_p = current_combined_count / prev_total
                        prev_p_nonzero = prev_p[prev_p > 0]
                        H_old = -np.sum(prev_p_nonzero * np.log2(prev_p_nonzero))
                    else:
                        H_old = 0.0
                    # 将该客户端加入当前集合，更新累计标签计数和熵列表
                    current_combined_count += self.label_counts[cid]
                    current_selected_entropies.append(self.local_entropies[cid])
                    # 计算加入该客户端后的全局熵 H_new 及熵增益
                    new_total = current_combined_count.sum()
                    if new_total > 0:
                        p_all = current_combined_count / new_total
                        p_all_nonzero = p_all[p_all > 0]
                        H_new = -np.sum(p_all_nonzero * np.log2(p_all_nonzero))
                    else:
                        H_new = 0.0
                    entropy_gain = float(H_new - H_old)
                    # 计算当前已选客户端熵值的方差及其惩罚项
                    entropy_var = float(np.var(current_selected_entropies))
                    entropy_var_penalty = float(alpha_vp * entropy_var)
                    # 计算数据量加权因子（未乘以 β）
                    data_factor = float(client.train_samples / self.avg_samples) if self.avg_samples > 0 else 1.0
                    # 计算该客户端的综合评分：score = H_new - α_vp * Var(H) + β * data_factor
                    score = float(H_new - entropy_var_penalty + beta * data_factor)
                    # 保存该客户端的选择过程指标
                    selection_metrics.append({
                        "client_id": cid,
                        "entropy_gain": entropy_gain,
                        "entropy_var_penalty": entropy_var_penalty,
                        "data_factor": data_factor,
                        "score": score
                    })
            # 7. 获取本轮缓冲区状态（FIFO队列中的客户端IDs列表）
            buffer_state = list(self.selection_buffer) if self.buffer_size > 0 else []

            # 8. 组装本轮的日志记录信息
            if self.log_enabled:
                round_log = {
                    "round": round_idx,
                    "selected_clients": selected_ids,
                    "clients": clients_info,
                    "global_label_distribution": global_label_distribution,
                    "global_entropy": global_entropy,
                    "selection_metrics": selection_metrics,
                    "buffer": buffer_state,
                    "performance": performance_metrics,
                    "round_time": float(time.time() - round_start_time)
                }
                # 写入该轮日志信息至文件
                self.log_file.write(json.dumps(round_log, ensure_ascii=False, indent=4))
                # 如果不是最后一轮，添加逗号和换行继续JSON数组格式
                if round_idx < self.global_rounds:
                    self.log_file.write(",\n")
                else:
                    self.log_file.write("\n")
                self.log_file.flush()  # 立即刷新写入

            # 在每轮结束后调用evaluate，以更新内部记录并输出当前性能（打印）
            self.evaluate(round_idx=round_idx)

        # 训练过程结束，记录结束时间和总耗时，并闭合日志JSON结构
        end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        total_time = time.time() - self.training_start_time
        if self.log_enabled:
            training_end_info = {"end_time": end_time_str, "total_time": float(total_time)}
            # 完善JSON文件结构：关闭 rounds 数组并写入 training_end 信息
            self.log_file.write("],\n" + json.dumps({"training_end": training_end_info}, ensure_ascii=False, indent=4)[1:] + "\n")
            self.log_file.close()
        # 保存训练结果（例如生成 .h5 文件，以便后续分析）
        self.save_results()

    def evaluate(self, round_idx=None, acc=None, loss=None):
        """
        评估当前全局模型性能，并记录训练/测试指标（用于保存和输出）。
        如果提供 acc 或 loss 列表，则将结果附加其中；否则附加到默认的内部列表。
        """
        # 获取测试集和训练集上的统计信息
        stats_test = self.test_metrics()
        stats_train = self.train_metrics()
        total_test_samples = sum(stats_test[1])
        total_train_samples = sum(stats_train[1])
        test_accuracy = sum(stats_test[2]) / total_test_samples if total_test_samples > 0 else 0.0
        test_auc = sum(stats_test[3]) / total_test_samples if total_test_samples > 0 else 0.0
        train_loss = sum(stats_train[2]) / total_train_samples if total_train_samples > 0 else 0.0
        # 存储指标供 save_results 使用
        if acc is None:
            self.rs_test_acc.append(test_accuracy)
        else:
            acc.append(test_accuracy)
        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)
        # 打印当前评估结果
        print(f"\n------------- Round number: {round_idx} -------------" if round_idx is not None else "\n------------- Evaluation -------------")
        print("Evaluate global model")
        print(f"Averaged Train Loss: {train_loss:.4f}")
        print(f"Averaged Test Accuracy: {test_accuracy:.4f}")
        print(f"Averaged Test AUC: {test_auc:.4f}")
        # 计算各客户端测试准确率/auc并打印其标准差
        accs = [correct/num for correct, num in zip(stats_test[2], stats_test[1]) if num > 0]
        aucs = [auc/num for auc, num in zip(stats_test[3], stats_test[1]) if num > 0]
        if accs:
            print(f"Std Test Accuracy: {np.std(accs):.4f}")
        if aucs:
            print(f"Std Test AUC: {np.std(aucs):.4f}")
