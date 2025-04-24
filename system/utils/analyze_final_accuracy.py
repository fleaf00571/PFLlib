import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Using pandas for easier statistics summary display

# --- Configuration ---
# Set your results folder path (e.g., './results/fashionmnist')
RESULT_DIR = '/home/ecs-user/projects/PFLlib/results/cifar100/dir0.1_0.2'  # MODIFY THIS PATH if needed


# Specify the key for test accuracy in the H5 files
ACC_KEY = 'rs_test_acc'

# Number of final rounds to analyze
NUM_ROUNDS_TO_ANALYZE = 50
# --- End Configuration ---

# Function to load results (same as before)
def load_all_results(result_dir, acc_key):
    """Loads specified accuracy data from all H5 files in the directory."""
    results = {}
    if not os.path.isdir(result_dir):
        print(f"Error: Directory not found: {result_dir}")
        return results

    print(f"Loading results from: {result_dir}")
    for filename in os.listdir(result_dir):
        if filename.endswith('.h5'):
            try:
                # --- Adjust method name extraction if needed ---
                method_name = filename.replace('.h5', '').split('_')[1]
            except IndexError:
                print(f"Warning: Could not extract method name from '{filename}'. Using full filename stem.")
                method_name = os.path.splitext(filename)[0]
            # --- End method name extraction ---

            file_path = os.path.join(result_dir, filename)
            try:
                with h5py.File(file_path, 'r') as f:
                    if acc_key in f:
                        data_array = np.array(f[acc_key])
                        if data_array.ndim == 0: # Handle scalar value if only one round exists
                             data_array = data_array.reshape(1,)
                        elif data_array.ndim > 1: # Flatten if necessary
                             data_array = data_array.flatten()

                        if data_array.size > 0:
                           results[method_name] = {acc_key: data_array}
                           print(f"  Loaded '{acc_key}' ({len(data_array)} rounds) for method '{method_name}' from {filename}")
                        else:
                           print(f"  Warning: Accuracy key '{acc_key}' in {filename} is empty. Skipping.")
                    else:
                        print(f"  Warning: Accuracy key '{acc_key}' not found in {filename}. Skipping.")
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    return results

# Function to analyze and plot the last N rounds with enhanced statistics
def analyze_and_plot_last_n_rounds(results, acc_key, num_rounds):
    """
    Analyzes the last N rounds of accuracy for each method, prints enhanced statistics
    (including std dev, range, IQR, CV), and generates a comparative plot.
    """
    analysis_summary = {}
    plot_data = {}
    stats_columns = [ # Define the order of columns for the summary table
        'Actual Rounds', 'Mean', 'Std Dev', 'Variance', 'Min',
        '25%', 'Median (50%)', '75%', 'Max', 'Range', 'IQR', 'CV (%)'
    ]


    print(f"\n--- Analyzing Last {num_rounds} Rounds ---")

    for method, data in results.items():
        if acc_key not in data:
            continue

        acc_values = data[acc_key]
        total_rounds = len(acc_values)

        if total_rounds == 0:
            print(f"Method '{method}': No accuracy data points found. Skipping.")
            analysis_summary[method] = {col: np.nan for col in stats_columns} # Fill with NaN
            analysis_summary[method]['Actual Rounds'] = 0
            continue

        actual_rounds_used = min(total_rounds, num_rounds)
        if actual_rounds_used < num_rounds:
            print(f"Method '{method}': Only {actual_rounds_used} total rounds available (less than requested {num_rounds}). Analyzing all available rounds.")

        last_n_acc = acc_values[-actual_rounds_used:]
        plot_data[method] = last_n_acc

        # Calculate statistics
        if actual_rounds_used > 0:
            mean_acc = np.mean(last_n_acc)
            std_dev_acc = np.std(last_n_acc) # 标准差
            var_acc = np.var(last_n_acc)     # 方差
            min_acc = np.min(last_n_acc)      # 最小值
            q1_acc = np.percentile(last_n_acc, 25) # 25%分位数
            median_acc = np.median(last_n_acc)     # 中位数 (50%分位数)
            q3_acc = np.percentile(last_n_acc, 75) # 75%分位数
            max_acc = np.max(last_n_acc)      # 最大值
            range_acc = max_acc - min_acc           # 范围/极差
            iqr_acc = q3_acc - q1_acc               # 四分位距
            # Coefficient of Variation (handle potential division by zero)
            cv_acc = (std_dev_acc / mean_acc) * 100 if mean_acc != 0 else np.nan # 变异系数 (%)

            stats = {
                'Actual Rounds': actual_rounds_used,
                'Mean': mean_acc,
                'Std Dev': std_dev_acc,      # 新增: 标准差
                'Variance': var_acc,
                'Min': min_acc,
                '25%': q1_acc,
                'Median (50%)': median_acc,
                '75%': q3_acc,
                'Max': max_acc,
                'Range': range_acc,          # 新增: 范围/极差
                'IQR': iqr_acc,              # 新增: 四分位距
                'CV (%)': cv_acc             # 新增: 变异系数 (%)
            }
            analysis_summary[method] = stats
        else:
             # Handle case where actual_rounds_used is 0 (shouldn't happen with outer check, but safe)
             analysis_summary[method] = {col: np.nan for col in stats_columns}
             analysis_summary[method]['Actual Rounds'] = 0


    # Print statistics using Pandas DataFrame
    if analysis_summary:
        print("\n--- Statistics Summary (Last {} Rounds or fewer) ---".format(num_rounds))
        # Create DataFrame with specified column order
        summary_df = pd.DataFrame.from_dict(analysis_summary, orient='index')[stats_columns]

        # Format floating point numbers for better readability
        float_cols = ['Mean', 'Std Dev', 'Variance', 'Min', '25%', 'Median (50%)',
                      '75%', 'Max', 'Range', 'IQR', 'CV (%)']
        for col in float_cols:
             # Apply formatting, handling potential NaN values
             summary_df[col] = summary_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

        summary_df['Actual Rounds'] = summary_df['Actual Rounds'].astype(int) # Ensure Actual Rounds is integer
        print(summary_df)
        print("\n---指标解释 (Indicator Explanation)---")
        print("Std Dev: 标准差，数值越大波动越大 (单位与准确率相同)")
        print("Variance: 方差，标准差的平方")
        print("Range: 极差 (Max - Min)，显示最大波动幅度")
        print("IQR: 四分位距 (75% - 25%)，衡量中间50%数据的波动范围，对异常值不敏感")
        print("CV (%): 变异系数 ((Std Dev / Mean) * 100)，相对波动大小，数值越小相对越稳定")
        print("---------------------------------------")
    else:
        print("\nNo data available for statistical summary.")

    # Plotting (remains the same)
    if plot_data:
        print("\nGenerating plot...")
        plt.figure(figsize=(14, 8))
        ax = plt.gca()

        for method, acc_data in plot_data.items():
            x_axis = np.arange(len(acc_data))
            ax.plot(x_axis, acc_data, label=f"{method} ({len(acc_data)} rounds)", alpha=0.8)

        ax.set_xlabel(f"Round Index within Last {num_rounds} Rounds (0 = Start of window)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy Trend Over Last {num_rounds} Rounds (or fewer if not available)")
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        print("Plot generated.")
    else:
        print("No data available to generate plot.")

# Main execution block
if __name__ == '__main__':
    # 1. Load data
    all_method_results = load_all_results(RESULT_DIR, ACC_KEY)

    if not all_method_results:
        print("\nExiting: No results were loaded.")
    else:
        # 2. Analyze last N rounds, print enhanced stats, and plot
        analyze_and_plot_last_n_rounds(all_method_results, ACC_KEY, NUM_ROUNDS_TO_ANALYZE)

        print("\nScript finished.")