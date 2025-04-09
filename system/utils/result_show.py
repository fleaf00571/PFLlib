# import os
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt

# # 设置你的结果文件夹路径（例如：'./results/'）
# RESULT_DIR = 'results'  # 修改为你的路径，例如 '/mnt/data/'

# # 自动读取所有 h5 文件
# def load_all_results(result_dir):
#     results = {}
#     for filename in os.listdir(result_dir):
#         if filename.endswith('.h5'):
#             method_name = filename.replace('.h5', '').split('_')[1]  # 提取方法名（如 FedAvg）
#             file_path = os.path.join(result_dir, filename)
#             with h5py.File(file_path, 'r') as f:
#                 data = {key: np.array(f[key]) for key in f.keys()}
#                 results[method_name] = data
#     return results

# # 可视化函数
# def plot_metrics(results):
#     # 提取可用的 metric 名称
#     acc_key = 'rs_test_acc'
#     loss_key = 'rs_train_loss'

#     # 创建准确率图
#     plt.figure(figsize=(12, 6))
#     for method, data in results.items():
#         if acc_key in data:
#             plt.plot(data[acc_key], label=method)
#     plt.title('Test Accuracy Comparison')
#     plt.xlabel('Rounds')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # 创建训练损失图
#     plt.figure(figsize=(12, 6))
#     for method, data in results.items():
#         if loss_key in data:
#             plt.plot(data[loss_key], label=method)
#     plt.title('Training Loss Comparison')
#     plt.xlabel('Rounds')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # 主程序
# if __name__ == '__main__':
#     results = load_all_results(RESULT_DIR)
#     plot_metrics(results)





import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# 设置你的结果文件夹路径（例如：'./results/'）
RESULT_DIR = 'results/'  # 修改为你的路径，例如 '/mnt/data/'

# 自动读取所有 h5 文件
def load_all_results(result_dir):
    results = {}
    for filename in os.listdir(result_dir):
        if filename.endswith('.h5'):
            method_name = filename
            file_path = os.path.join(result_dir, filename)
            with h5py.File(file_path, 'r') as f:
                data = {key: np.array(f[key]) for key in f.keys()}
                results[method_name] = data
    return results

# 可视化函数（使用子图 subplots）
def plot_metrics(results):
    acc_key = 'rs_test_acc'
    loss_key = 'rs_train_loss'

    # 创建子图 (2行1列)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 绘制测试准确率对比
    for method, data in results.items():
        if acc_key in data:
            axes[0].plot(data[acc_key], label=method)
    axes[0].set_title('Test Accuracy Comparison')
    axes[0].set_xlabel('Rounds')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # 绘制训练损失对比
    for method, data in results.items():
        if loss_key in data:
            axes[1].plot(data[loss_key], label=method)
    axes[1].set_title('Training Loss Comparison')
    axes[1].set_xlabel('Rounds')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == '__main__':
    results = load_all_results(RESULT_DIR)
    plot_metrics(results)
