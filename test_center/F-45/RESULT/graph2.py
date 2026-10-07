import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 读取单个文件的数据，并提取所需列
def process_file(filepath):
    try:
        data = np.loadtxt(filepath, delimiter=',')  # 假设文件以逗号分隔
        col1 = data[:, 0]  # 第一列数据
        col2 = data[:, 1]  # 第二列数据
        col3 = data[:, 2]  # 第三列数据
        col4 = data[:, 3]  # 第四列数据
        col5 = data[:, 4]  # 第五列数据
        return col1, col2, col3, col4, col5
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None, None, None, None

# 绘制小图（散点图）
def plot_inset(ax, col2, col4, col5):
    ax.scatter(col2, col4, color='green', alpha=0.5, marker='o', label='Average')
    ax.scatter(col2, col5, color='green', alpha=0.5, marker='^', label='Layer')
    ax.set_title('b', fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(fontsize=8)

# 主程序
def main():
    # 小图文件
    inset_filename = "non_uniform_8_2.txt"
    inset_col1, inset_col2, inset_col3, inset_col4, inset_col5 = process_file(inset_filename)

    if any(x is None for x in (inset_col2, inset_col4, inset_col5)):
        print("Error in processing files.")
        return

    # 创建小图
    fig, ax = plt.subplots(figsize=(6, 5))  # 可以根据需要调整大小
    plot_inset(ax, inset_col2, inset_col4, inset_col5)

    # 保存小图
    plt.savefig('inset_plot.png', dpi=300, bbox_inches='tight')
    print("Inset plot saved as inset_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
