import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体和字号
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18  # 设置全局字号为18号

# 生成文件名列表
def generate_filenames():
    filenames = []
    for i in range(2, 21):
        filenames.append(f"{i}_10.txt")
    return filenames

# 读取单个文件的数据，并提取所需列
def process_file(filepath):
    try:
        data = np.loadtxt(filepath, delimiter=',')  # 假设文件以逗号分隔
        col1 = data[:, 0]  # 第一列数据
        col2 = data[:, 1]  # 第二列数据
        col4 = data[:, 3]  # 第四列数据
        return col1, col2, col4
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None, None

# 主程序
def main():
    filenames = generate_filenames()

    # 主图数据
    all_col1 = []
    all_col2_mean = []
    all_col2_std = []
    all_col4_mean = []
    all_col4_std = []

    # 小图数据（全局误差）
    global_errors = []

    for filename in filenames:
        # 处理每个文件
        col1, col2, col4 = process_file(filename)
        if col1 is not None and col2 is not None and col4 is not None:
            col1_x = np.mean(col1)  # 每个文件的 `col1` 平均值代表 x 轴
            col2_mean = np.mean(col2)
            col2_std = np.std(col2)
            col4_mean = np.mean(col4)
            col4_std = np.std(col4)

            # 主图数据
            all_col1.append(col1_x)
            all_col2_mean.append(col2_mean)
            all_col2_std.append(col2_std)
            all_col4_mean.append(col4_mean)
            all_col4_std.append(col4_std)

            # 计算当前文件的全局误差（逐行差值的标准差）
            file_errors = col2 - col4  # 按行计算误差
            global_error = np.std(file_errors)  # 全局误差的标准差
            global_errors.append(global_error)

    # 转为数组方便处理
    all_col1 = np.array(all_col1)
    all_col2_mean = np.array(all_col2_mean)
    all_col2_std = np.array(all_col2_std)
    all_col4_mean = np.array(all_col4_mean)
    all_col4_std = np.array(all_col4_std)
    global_errors = np.array(global_errors)

    # 按 `col1` 排序，确保绘制的折线平滑
    sorted_indices = np.argsort(all_col1)
    all_col1 = all_col1[sorted_indices]
    all_col2_mean = all_col2_mean[sorted_indices]
    all_col2_std = all_col2_std[sorted_indices]
    all_col4_mean = all_col4_mean[sorted_indices]
    all_col4_std = all_col4_std[sorted_indices]
    global_errors = global_errors[sorted_indices]

    # 主图绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(all_col1, all_col2_mean, label="Real", color='blue', linewidth=2)
    ax.fill_between(all_col1, all_col2_mean - all_col2_std, all_col2_mean + all_col2_std, color='blue', alpha=0.2)
    ax.plot(all_col1, all_col4_mean, label="Approximate", color='red', linewidth=2)
    ax.fill_between(all_col1, all_col4_mean - all_col4_std, all_col4_mean + all_col4_std, color='red', alpha=0.2)

    # 图例调整到左上角
    ax.legend(loc='upper left', fontsize=18)
    ax.set_title("ER", fontsize=18)
    ax.set_xlabel("The Number of Hyper Edge", fontsize=18)
    ax.set_ylabel("The Smallest Nonzero Eigenvalue", fontsize=18)
    # ax.grid(True, linestyle='--', alpha=0.6)
    
    # 设置x轴和y轴范围
    ax.set_xlim(2, 20)  # 设置x轴范围
    ax.set_ylim(0, None)  # 设置y轴范围，None表示正无穷
    
    # 取消网格线
    ax.grid(False)

    # 小图绘制
    inset_ax = fig.add_axes([0.63, 0.6, 0.3, 0.3])  # 定义小图的位置 [left, bottom, width, height]
    inset_ax.scatter(all_col1, global_errors, color='green', label="Global Error", s=50, alpha=0.7)
    # inset_ax.set_title("Global Errors", fontsize=10)
    # inset_ax.set_xlabel("Col1", fontsize=8)
    # inset_ax.set_ylabel("Error", fontsize=8)
    # inset_ax.tick_params(axis='both', which='major', labelsize=8)
    # inset_ax.grid(True, linestyle='--', alpha=0.6)
    inset_ax.legend(fontsize=14)

    # 保存并显示
    plt.tight_layout()
    plt.savefig("line_plot_with_global_error_inset.png")
    plt.show()

if __name__ == "__main__":
    main()
