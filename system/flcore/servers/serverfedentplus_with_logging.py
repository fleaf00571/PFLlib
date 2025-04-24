import time
import json
import os
import numpy as np
from collections import deque
from flcore.servers.serverfedentplus import FedEntPlus
from utils.data_utils import read_client_data

class FedEntPlusWithLogging(FedEntPlus):
    def __init__(self, args, times):
        super().__init__(args, times)  # 调用父类初始化（FedEntPlus），建立客户端列表等
        # 确定日志文件路径和名称，可包含算法名称和运行编号times
        # 日志文件路径和名称
        log_filename = f"{args.save_folder_name}/{args.dataset}_{args.algorithm}_run{times}_log.json"
        
        # 检查并创建文件夹（若不存在）
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        
        # 打开文件写入
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

        # 将训练起始信息写入日志文件
        init_json = json.dumps({"training_start": training_start_info}, ensure_ascii=False, indent=4)
        self.log_file.write(init_json[:-1] + ',\n  "rounds": [\n')


        # 预先计算所有客户端的原始标签分布（在未加噪前）用于日志记录
        self.raw_label_counts = []
        for cid in range(self.num_clients):
            # 读取客户端 cid 的训练数据并统计标签计数
            label_count = np.zeros(self.num_classes, dtype=int)
            train_data = read_client_data(self.dataset, cid, is_train=True)
            for _, y in train_data:
                label = int(y.item()) if hasattr(y, "item") else int(y)
                if 0 <= label < self.num_classes:
                    label_count[label] += 1
            self.raw_label_counts.append(label_count.tolist())  # 存储为列表方便序列化

        # 记录训练起始时间戳（秒）用于计算总耗时
        self.training_start_time = time.time()

    def train(self):
        """运行联邦训练并记录日志。"""
        # 遍历全局训练轮次
        for round_idx in range(1, self.global_rounds + 1):
            round_start_time = time.time()
            # 1. 选择本轮参与客户端
            selected_clients = self.select_clients()  # 使用父类的选择算法
            self.selected_clients = selected_clients  # 保存选中客户端列表供后续步骤使用
            selected_ids = [c.id for c in selected_clients]

            # 2. 准备记录选中客户端的详细信息
            clients_info = {}
            for client in selected_clients:
                cid = client.id
                clients_info[cid] = {
                    "raw_label_distribution": self.raw_label_counts[cid],  # 原始标签分布
                    "dp_label_distribution": self.label_counts[cid].tolist(),  # 加噪后的标签分布
                    "local_entropy": float(self.local_entropies[cid]),  # 本地标签熵值
                    "train_samples": int(client.train_samples)  # 训练样本数
                }

            # 3. 下发全局模型并进行本地训练
            self.send_models()  # 将当前全局模型发送给所有客户端
            for client in self.selected_clients:
                client.train()  # 客户端进行本地训练
            self.receive_models()      # 接收客户端模型更新
            self.aggregate_parameters()  # 聚合更新全局模型

            # 4. 评估全局模型在测试集上的性能（准确率、AUC）以及训练损失
            # 将新全局模型下发所有客户端以确保同步，然后测评
            self.send_models()
            ids, num_samples, tot_correct, tot_auc = self.test_metrics()
            _, num_train_samples, train_losses = self.train_metrics()
            # 计算总体测试准确率、AUC和平均训练损失
            total_samples = sum(num_samples)
            test_accuracy = float(sum(tot_correct) / total_samples if total_samples > 0 else 0.0)
            test_auc = float(sum(tot_auc) / total_samples if total_samples > 0 else 0.0)
            avg_train_loss = float(sum(train_losses) / sum(num_train_samples) if sum(num_train_samples) > 0 else 0.0)

            performance_metrics = {
                "test_accuracy": test_accuracy,
                "test_auc": test_auc,
                "train_loss": avg_train_loss
            }

            # 5. 计算本轮合并后的全局标签分布及熵值
            # 使用选中客户端的标签计数（加噪后）求和得到全局分布
            combined_count = np.zeros(self.num_classes, dtype=float)
            for cid in selected_ids:
                combined_count += self.label_counts[cid]
            global_label_distribution = combined_count.tolist()
            # 计算全局熵值
            total_count = combined_count.sum()
            if total_count > 0:
                p = combined_count / total_count
                p_nonzero = p[p > 0]
                global_entropy = float(-np.sum(p_nonzero * np.log2(p_nonzero)))
            else:
                global_entropy = 0.0

            # 6. 记录客户端选择过程的评分指标（按选择顺序）
            selection_metrics = []
            # 准备迭代计算每次选中时的增益和评分
            current_combined_count = np.zeros(self.num_classes, dtype=float)
            current_selected_entropies = []
            # α_vp 和 β 超参数
            alpha_vp = getattr(self.args, "alpha_vp", 0.0)
            beta = getattr(self.args, "beta", 0.0)
            for idx, client in enumerate(selected_clients):
                cid = client.id
                # 计算选中该客户端前后的全局熵变化（熵增益）
                prev_total = current_combined_count.sum()
                if prev_total > 0:
                    prev_p = current_combined_count / prev_total
                    prev_p_nonzero = prev_p[prev_p > 0]
                    H_old = -np.sum(prev_p_nonzero * np.log2(prev_p_nonzero))
                else:
                    H_old = 0.0
                # 更新累计的标签计数和熵列表
                current_combined_count += self.label_counts[cid]
                current_selected_entropies.append(self.local_entropies[cid])
                # 计算加入该客户端后的全局熵和熵增益
                new_total = current_combined_count.sum()
                p_all = current_combined_count / new_total if new_total > 0 else np.zeros(self.num_classes)
                p_all_nonzero = p_all[p_all > 0]
                H_new = -np.sum(p_all_nonzero * np.log2(p_all_nonzero)) if new_total > 0 else 0.0
                entropy_gain = float(H_new - H_old)
                # 计算当前选中集合的熵值方差及惩罚项
                entropy_var = float(np.var(current_selected_entropies))
                entropy_var_penalty = float(alpha_vp * entropy_var)
                # 计算数据量加权因子（未乘 β）以及综合评分
                data_factor = float(client.train_samples / self.avg_samples if self.avg_samples > 0 else 1.0)
                score = float(H_new - entropy_var_penalty + beta * data_factor)
                # 保存该客户端的选择指标
                selection_metrics.append({
                    "client_id": cid,
                    "entropy_gain": entropy_gain,
                    "entropy_var_penalty": entropy_var_penalty,
                    "data_factor": data_factor,
                    "score": score
                })

            # 7. 获取本轮缓冲区状态（FIFO队列中的客户端IDs）
            buffer_state = list(self.selection_buffer)

            # 8. 组装本轮日志信息
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
            # 将该轮日志写入文件（添加逗号以继续JSON数组格式，最后一轮在训练结束时处理）
            self.log_file.write(json.dumps(round_log, ensure_ascii=False, indent=4))
            # 如果不是最后一轮，添加逗号和换行分隔
            if round_idx < self.global_rounds:
                self.log_file.write(",\n")
            else:
                self.log_file.write("\n")
            self.log_file.flush()  # 立即刷新到文件

            # 新增调用evaluate，确保更新 rs_test_acc, rs_test_auc, rs_train_loss
            self.evaluate(round_idx=round_idx)

            # （可选）检测提前终止条件
            if getattr(self.args, "auto_break", False):
                # 如果启用了 auto_break，可以根据需要的条件中断训练，例如准确率收敛等
                # 这里简单示例：如果达到目标精度则提前停止
                target_acc = getattr(self.args, "target_accuracy", None)
                if target_acc and test_accuracy >= target_acc:
                    # 将剩余轮次跳过
                    break

        # 训练循环结束后，记录训练结束时间和总耗时
        end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        total_time = time.time() - self.training_start_time
        training_end_info = {
            "end_time": end_time_str,
            "total_time": float(total_time)
        }
        # 完善JSON文件闭合结构，写入 training_end 并关闭大括号
        self.log_file.write("],\n" + json.dumps({"training_end": training_end_info}, ensure_ascii=False, indent=4)[1:] + "\n")
        # 上行中 json.dumps(...)[1:] 去掉生成的第一个 '{'，以匹配现有JSON结构正确闭合
        # 完整文件结构将形成:
        # {
        #   "training_start": {...},
        #   "rounds": [
        #       { ... }, 
        #       { ... }
        #   ],
        #   "training_end": {...}
        # }
        self.log_file.close()


        # 新增：调用父类内置方法，保存标准 .h5 文件
        self.save_results()  # ← 添加这一行，产生PFLlib所需的 .h5 结果文件


    def evaluate(self, round_idx=None, acc=None, loss=None):
        # 获取测试和训练数据统计
        stats_test = self.test_metrics()
        stats_train = self.train_metrics()

        # 计算全局指标
        total_test_samples = sum(stats_test[1])
        total_train_samples = sum(stats_train[1])
        test_accuracy = sum(stats_test[2]) / total_test_samples
        test_auc = sum(stats_test[3]) / total_test_samples
        train_loss = sum(stats_train[2]) / total_train_samples

        # 存储指标（用于后续HDF5文件保存）
        if acc is None:
            self.rs_test_acc.append(test_accuracy)
        else:
            acc.append(test_accuracy)

        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        # 打印美观、详细的评价信息（参照FedAvg原风格）
        print(f"\n------------- Round number: {round_idx} -------------" if round_idx is not None else "\n------------- Evaluation -------------")
        print("\nEvaluate global model")
        print(f"Averaged Train Loss: {train_loss:.4f}")
        print(f"Averaged Test Accuracy: {test_accuracy:.4f}")
        print(f"Averaged Test AUC: {test_auc:.4f}")

        # 计算并打印标准差
        accs = [correct / num for correct, num in zip(stats_test[2], stats_test[1])]
        aucs = [auc / num for auc, num in zip(stats_test[3], stats_test[1])]
        print(f"Std Test Accuracy: {np.std(accs):.4f}")
        print(f"Std Test AUC: {np.std(aucs):.4f}")