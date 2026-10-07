import numpy as np
import matplotlib.pyplot as plt

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

    plt.figure(figsize=(10, 6))

    all_col1 = []
    all_col2_mean = []
    all_col2_std = []
    all_col4_mean = []
    all_col4_std = []

    for filename in filenames:
        # 处理每个文件
        col1, col2, col4 = process_file(filename)
        if col1 is not None and col2 is not None and col4 is not None:
            col1_x = np.mean(col1)  # 每个文件的 `col1` 平均值代表 x 轴
            col2_mean = np.mean(col2)
            col2_std = np.std(col2)
            col4_mean = np.mean(col4)
            col4_std = np.std(col4)

            # 保存数据
            all_col1.append(col1_x)
            all_col2_mean.append(col2_mean)
            all_col2_std.append(col2_std)
            all_col4_mean.append(col4_mean)
            all_col4_std.append(col4_std)

    # 转为数组方便处理
    all_col1 = np.array(all_col1)
    all_col2_mean = np.array(all_col2_mean)
    all_col2_std = np.array(all_col2_std)
    all_col4_mean = np.array(all_col4_mean)
    all_col4_std = np.array(all_col4_std)

    # 按 `col1` 排序，确保绘制的折线平滑
    sorted_indices = np.argsort(all_col1)
    all_col1 = all_col1[sorted_indices]
    all_col2_mean = all_col2_mean[sorted_indices]
    all_col2_std = all_col2_std[sorted_indices]
    all_col4_mean = all_col4_mean[sorted_indices]
    all_col4_std = all_col4_std[sorted_indices]

    # 绘制 col2 曲线和误差范围
    plt.plot(all_col1, all_col2_mean, label="Col 2 Mean", color='blue', linewidth=2)
    plt.fill_between(all_col1, all_col2_mean - all_col2_std, all_col2_mean + all_col2_std, color='blue', alpha=0.2)

    # 绘制 col4 曲线和误差范围
    plt.plot(all_col1, all_col4_mean, label="Col 4 Mean", color='red', linewidth=2)
    plt.fill_between(all_col1, all_col4_mean - all_col4_std, all_col4_mean + all_col4_std, color='red', alpha=0.2)

    # 图例和标签
    plt.title("Line Plot with Error Fill", fontsize=14)
    plt.xlabel("Col1 (X-axis)", fontsize=12)
    plt.ylabel("Values (Y-axis)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存并显示
    plt.tight_layout()
    plt.savefig("line_plot_with_error_fill.png")
    plt.show()

if __name__ == "__main__":
    main()
