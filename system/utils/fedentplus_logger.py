import json
import os

class FedEntPlusLogger:
    def __init__(self, log_file='fedentplus_log.json'):
        self.log_file = log_file
        self.log_data = {
            "experiment_config": {},
            "rounds": []
        }
    
    def log_experiment_config(self, config):
        """记录实验的初始配置"""
        self.log_data["experiment_config"] = config

    def log_round(self, round_number, client_selection, metrics, additional_data=None):
        """记录每轮训练的详细信息"""
        round_entry = {
            "round": round_number,
            "client_selection": client_selection,
            "metrics": metrics
        }
        if additional_data:
            round_entry["additional"] = additional_data
        self.log_data["rounds"].append(round_entry)

    def save(self):
        """将日志数据写入 JSON 文件"""
        folder = os.path.dirname(self.log_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=4)
