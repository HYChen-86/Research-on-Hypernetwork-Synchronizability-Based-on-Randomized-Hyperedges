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

    for filename in filenames:
        # 处理每个文件
        col1, col2, col4 = process_file(filename)
        if col1 is not None and col2 is not None and col4 is not None:
            # 计算每列数据的平均值和标准差
            col1_x = np.mean(col1)  # 每个文件的 `col1` 代表一个点的 x 轴
            col2_mean = np.mean(col2)
            col2_std = np.std(col2)
            col4_mean = np.mean(col4)
            col4_std = np.std(col4)

            # 绘制 col2 的平均值及误差棒
            plt.errorbar(
                col1_x, col2_mean, yerr=col2_std, fmt='o', color='blue', 
                ecolor='lightblue', elinewidth=2, capsize=4, label="Col 2 Mean" if filename == filenames[0] else ""
            )

            # 绘制 col4 的平均值及误差棒
            plt.errorbar(
                col1_x, col4_mean, yerr=col4_std, fmt='o', color='red', 
                ecolor='lightcoral', elinewidth=2, capsize=4, label="Col 4 Mean" if filename == filenames[0] else ""
            )

    # 图例和标签
    plt.title("Average and Error Bars per File", fontsize=14)
    plt.xlabel("Col1 (X-axis)", fontsize=12)
    plt.ylabel("Values (Y-axis)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存并显示
    plt.tight_layout()
    plt.savefig("col1_per_file_error_bars.png")
    plt.show()

if __name__ == "__main__":
    main()
