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

# 绘制主图（散点图）
def plot_scatter(ax, col1, col2, col3, col4, col5):
    if any(x is None for x in (col1, col2, col3, col4, col5)):
        print("Invalid data for plotting.")
        return

    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["lightblue", "blue"])
    norm = plt.Normalize(1, 7)
    point_size = 100

    scatter3 = ax.scatter(col1, col3, c=col2, cmap=cmap, norm=norm, marker='D', s=point_size, label='Multi')
    scatter4 = ax.scatter(col1, col4, c=col2, cmap=cmap, norm=norm, marker='o', s=point_size, label='Average')
    scatter5 = ax.scatter(col1, col5, c=col2, cmap=cmap, norm=norm, marker='^', s=point_size, label='Layer')

    cbar = plt.colorbar(scatter5, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('The Number of New_Node', fontsize=12)

    ax.set_xlabel('The Number of Hyper Edge', fontsize=12)
    ax.set_ylabel('The Smallest Nonzero Eigenvalue  (λ_2)', fontsize=12)
    ax.set_title('Scatter Plot of Columns', fontsize=14)
    ax.legend()

# 绘制小图（散点图）
def plot_inset(ax, col2, col4, col5):
    ax.scatter(col2, col4, color='green', alpha=0.5, marker='o', label='Average')
    ax.scatter(col2, col5, color='green', alpha=0.5, marker='^', label='Layer')
    # ax.set_xlabel('Column 2', fontsize=8)
    # ax.set_ylabel('Values', fontsize=8)
    ax.set_title('b', fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(fontsize=8)

# 主程序
def main():
    # 主图文件
    main_filename = "non_uniform_8.txt"
    col1, col2, col3, col4, col5 = process_file(main_filename)

    # 小图文件
    inset_filename = "non_uniform_8_2.txt"
    inset_col1, inset_col2, inset_col3, inset_col4, inset_col5 = process_file(inset_filename)

    if any(x is None for x in (col1, col2, col3, col4, col5, inset_col2)):
        print("Error in processing files.")
        return

    # 创建主图和小图
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_scatter(ax, col1, col2, col3, col4, col5)

    # 添加小图
    inset_ax = inset_axes(ax, width="40%", height="30%", loc='upper right', borderpad=2)
    plot_inset(inset_ax, inset_col2, inset_col3, inset_col4)

    # 保存图片
    plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as b.png")
    plt.show()

if __name__ == "__main__":
    main()
