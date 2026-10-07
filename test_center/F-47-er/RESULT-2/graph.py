import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体和字号
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18  # 设置全局字号为18号

# 生成文件名列表
def generate_filenames():
    filenames = []
    for i in range(14, 62, 4):
        filenames.append(f"40_5_{i}.txt")
    return filenames

# 读取单个文件的数据，并提取所需列
def process_file(filepath):
    try:
        data = np.loadtxt(filepath, delimiter=',')  # 假设文件以逗号分隔
        x_values = data[:, 2]  # 第三列数据
        y_values = data[:, 1]  # 第二列数据
        return x_values, y_values
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None

# 主程序
def main():
    filenames = generate_filenames()
    all_x = []
    all_y = []

    for filename in filenames:
        x_values, y_values = process_file(filename)
        if x_values is not None and y_values is not None:
            all_x.extend(x_values)
            all_y.extend(y_values)

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(all_x, all_y, alpha=0.7, edgecolors='w', s=60)
    plt.title("ER", fontsize=18)
    plt.xlabel("The Average Node_Number of Hyper Edge", fontsize=18)
    plt.ylabel("The Smallest Nonzero Eigenvalue  (λ_2)", fontsize=18)
    plt.xlim()  # 横轴范围
    plt.ylim()  # 纵轴从 0 开始
    # plt.grid(True, linestyle='--', alpha=0.6)

    # 保存并显示
    plt.tight_layout()
    plt.savefig("result_plot_no_line_connecting_max_min.png", dpi=100)
    plt.show()

if __name__ == "__main__":
    main()
