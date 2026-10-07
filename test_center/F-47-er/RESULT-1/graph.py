import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 设置全局字体和字号
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18  # 设置全局字号为18号

# 生成文件名列表
def generate_filenames():
    filenames = []
    for base in [20, 40, 60, 80, 100, 120]:
        for i in range(1, 7):
            filenames.append(f"{base}_{i}_10.txt")
    return filenames

# 读取单个文件的数据，并计算平均值、最大值和最小值
def process_file(filepath):
    try:
        data = np.loadtxt(filepath, delimiter=',')
        x_values = data[:, 0]
        y_values = data[:, 1]
        x_mean = np.mean(x_values)
        y_mean = np.mean(y_values)
        y_min = np.min(y_values)
        y_max = np.max(y_values)
        return x_mean, y_mean, y_min, y_max
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None, None, None

# 生成浅色
def lighten_color(color, amount=0.5):
    try:
        c = mcolors.to_rgba(color)
        return mcolors.to_rgba((1 - amount) * np.array(c[:3]) + amount * np.array([1, 1, 1]), c[3])
    except ValueError:
        print(f"Could not lighten color: {color}")
        return color

# 主函数
def main():
    filenames = generate_filenames()
    data_by_group = {}

    # 处理文件并按照组分类
    for filename in filenames:
        group = filename.split('_')[1]  # 获取组号
        x_mean, y_mean, y_min, y_max = process_file(filename)

        if x_mean is not None and y_mean is not None:
            if group not in data_by_group:
                data_by_group[group] = []
            data_by_group[group].append((x_mean, y_mean, y_min, y_max))

    # 手动设置六个颜色
    custom_colors = [
        '#1f77b4',  # 蓝色
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b'   # 棕色
    ]

    # 定义不同的点形状
    # markers = ['o', 's', 'D', 'P', '^', '*']  # 圆形、正方形、菱形、五边形、三角形、星形
    markers = ['o', 'o', 'o', 'o', 'o', 'o']

    # 绘制曲线
    plt.figure(figsize=(10, 8))

    for idx, (group, data) in enumerate(data_by_group.items()):
        data.sort(key=lambda x: x[0])  # 按横坐标排序
        x_values = [item[0] for item in data]
        y_values = [item[1] for item in data]
        y_min_values = [item[2] for item in data]
        y_max_values = [item[3] for item in data]

        # 获取手动设置的颜色
        base_color = custom_colors[idx % len(custom_colors)]  # 循环使用颜色
        light_color = lighten_color(base_color, 0.6)  # 生成浅色
        marker = markers[idx % len(markers)]  # 确保循环使用标记

        # 绘制平均值曲线（实线）
        plt.plot(x_values, y_values, marker=marker, color=base_color, markersize=8, label=f"old_node={group}", linewidth=3)

        # 绘制最大值和最小值点，颜色为浅色，形状与曲线相同
        plt.scatter(x_values, y_max_values, marker='_', color=light_color, s=100,)
        plt.scatter(x_values, y_min_values, marker='_', color=light_color, s=100,)

        # 只绘制最大值和最小值之间的虚线，不进行任何实线连接
        for x, y_min, y_max in zip(x_values, y_min_values, y_max_values):
            plt.vlines(x, ymin=y_min, ymax=y_max, color=light_color, linestyle='-', linewidth=3)

    # 图形美化
    plt.title("ER", fontsize=18, fontweight='bold', family='Times New Roman')
    plt.xlabel("The Number of Hyper Edge", fontsize=18, fontweight='bold', family='Times New Roman')
    plt.ylabel("The Smallest Nonzero Eigenvalue  (λ_2)", fontsize=18, fontweight='bold', family='Times New Roman')
    plt.xlim(10, 130)  # 横轴范围
    plt.ylim(0, None)  # 纵轴从 0 开始
    # plt.grid()

    # 调整图例位置和排列方式
    plt.legend(loc='upper right', ncol=3, fontsize=18, frameon=True)

    # 保存并显示
    plt.tight_layout()
    plt.savefig("result_plot_no_line_connecting_max_min.png", dpi=300)  # 设置 dpi=300 或更高以提高分辨率
    plt.show()

if __name__ == "__main__":
    main()
