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
        col2 = data[:, 1]  # 第二列数据
        col4 = data[:, 3]  # 第四列数据
        return col2, col4
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None

# 主程序
def main():
    filenames = generate_filenames()
    col2_values = []
    col4_values = []

    # 收集数据
    for filename in filenames:
        col2, col4 = process_file(filename)
        if col2 is not None and col4 is not None:
            col2_values.append(col2)
            col4_values.append(col4)

    # 转为数组以便处理
    col2_values = np.array(col2_values)
    col4_values = np.array(col4_values)

    # 计算平均值和标准误差
    col2_mean = np.mean(col2_values, axis=0)
    col2_std = np.std(col2_values, axis=0)
    col4_mean = np.mean(col4_values, axis=0)
    col4_std = np.std(col4_values, axis=0)

    x = np.arange(len(col2_mean))  # 假设每列对应的 x 轴索引

    # 全局误差
    col2_global_std = np.mean(col2_std)
    col4_global_std = np.mean(col4_std)

    # 绘制主图
    plt.figure(figsize=(8, 6))

    # 绘制第二列平均值曲线及误差范围
    plt.plot(x, col2_mean, label="Parental", color='blue', linewidth=2)
    plt.fill_between(x, col2_mean - col2_std, col2_mean + col2_std, color='blue', alpha=0.2)

    # 绘制第四列平均值曲线及误差范围
    plt.plot(x, col4_mean, label="Δbfd2", color='red', linewidth=2)
    plt.fill_between(x, col4_mean - col4_std, col4_mean + col4_std, color='red', alpha=0.2)

    # 图例和标签
    plt.title("Infectivity Over Time", fontsize=14)
    plt.xlabel("Time extracellular (h)", fontsize=12)
    plt.ylabel("Infectivity (%)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # # 添加小图
    # inset_axes = plt.axes([0.6, 0.6, 0.3, 0.3])  # [x, y, width, height]
    # bar_labels = ['Col 2', 'Col 4']
    # bar_values = [col2_global_std, col4_global_std]
    # colors = ['blue', 'red']

    # inset_axes.bar(bar_labels, bar_values, color=colors, alpha=0.7)
    # inset_axes.set_title("Global Error", fontsize=10)
    # inset_axes.set_ylabel("Std Dev", fontsize=8)
    # inset_axes.tick_params(axis='both', labelsize=8)

    # 保存并显示
    plt.tight_layout()
    plt.savefig("infectivity_with_error_ranges_and_inset.png")
    plt.show()

if __name__ == "__main__":
    main()
