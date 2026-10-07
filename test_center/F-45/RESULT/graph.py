import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 生成文件名列表
def generate_filenames():
    filenames = []
    # for i in range(2, 21):
    filenames.append(f"non_uniform_8.txt")
    return filenames

# 读取单个文件的数据，并提取所需列
def process_file(filepath):
    try:
        data = np.loadtxt(filepath, delimiter=',')  # 假设文件以逗号分隔
        col1 = data[:, 0]  # 第一列数据
        col2 = data[:, 1]  # 第二列数据
        col3 = data[:, 2]  # 第二列数据
        col4 = data[:, 3]  # 第四列数据
        col5 = data[:, 4]  # 第五列数据
        return col1, col2, col3, col4, col5
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None, None, None, None
    
# 绘制散点图
def plot_scatter(col1, col2, col3, col4, col5, save_path='scatter_plot_1.png'):
    # 检查是否有无效数据
    if any(x is None for x in (col1, col2, col3, col4, col5)):
        print("Invalid data for plotting.")
        return
    
    # 创建渐变颜色映射
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["lightblue", "blue"])
    norm = plt.Normalize(1, 7)  # 假设col2的范围是1到7

    # 点大小设定（可以根据实际需求修改）
    point_size = 100  # 固定大小
    # 也可以用以下方式根据某列动态调整大小
    # point_size_col3 = col3 * 10  # 根据col3放大，需确保col3值非负
    # point_size_col4 = col4 * 10
    # point_size_col5 = col5 * 10
    
    # 创建图形（保持长宽比为1:1）
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制第三列（用菱形表示）
    scatter3 = ax.scatter(col1, col3, c=col2, cmap=cmap, norm=norm, marker='D', s=point_size, label='AVE')

    # 绘制第四列（用三角形表示）
    scatter4 = ax.scatter(col1, col4, c=col2, cmap=cmap, norm=norm, marker='o', s=point_size, label='CAL')
    
    # 绘制第五列（用圆形表示）
    scatter5 = ax.scatter(col1, col5, c=col2, cmap=cmap, norm=norm, marker='^', s=point_size, label='REAL')
    
    # 添加颜色条（设置在右侧）
    cbar = plt.colorbar(scatter5, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('The Number of New_Node', fontsize=12)
    
    # 图表设置
    ax.set_xlabel('The Number of Hyper Edge', fontsize=12)
    ax.set_ylabel('The Smallest Nonzero Eigenvalue  (λ_2)', fontsize=12)
    ax.set_title('Scatter Plot of Columns', fontsize=14)
    ax.legend()
    # ax.grid(True)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {save_path}")
    plt.show()

# 主程序
filenames = generate_filenames()
for filename in filenames:
    col1, col2, col3, col4, col5 = process_file(filename)
    plot_scatter(col1, col2, col3, col4, col5)