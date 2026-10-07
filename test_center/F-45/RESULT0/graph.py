import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 设置全局字体和字号
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18  # 设置全局字号为18号

# 生成文件名列表
def generate_filenames():
    filenames = []
    filenames.append("uniform_8.txt")
    return filenames

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

# 对数据进行分组并计算平均值
def calculate_averages(col2, col3, col4, col5):
    unique_vals = np.unique(col2)
    avg_col3, avg_col4, avg_col5 = [], [], []
    for val in unique_vals:
        indices = np.where(col2 == val)
        avg_col3.append(np.mean(col3[indices]))
        avg_col4.append(np.mean(col4[indices]))
        avg_col5.append(np.mean(col5[indices]))
    return unique_vals, avg_col3, avg_col4, avg_col5

# 绘制平滑曲线
def plot_data(x, y1, y2, y3):
    # Increase the number of points for smoother interpolation
    x_smooth = np.linspace(min(x), max(x), 500)  # Increased from 300 to 500 for finer smoothness
    y1_smooth = make_interp_spline(x, y1, k=3)(x_smooth)  # k=3 for cubic spline (default, but adjustable)
    y2_smooth = make_interp_spline(x, y2, k=3)(x_smooth)
    y3_smooth = make_interp_spline(x, y3, k=3)(x_smooth)

    # Clamp the blue line (y1) between the green line (y2) and the red line (y3)
    y1_smooth = np.clip(y1_smooth, y2_smooth, y3_smooth)

    plt.figure(figsize=(10, 8))
    # plt.plot(x_smooth, y1_smooth, label='Column 3 Avg', color='blue')
    plt.plot(x_smooth, y2_smooth, label='Real', color='green', linewidth=3)
    plt.plot(x_smooth, y3_smooth, label='Avg', color='red', linewidth=3)
    plt.xlabel('The Number of N_old')
    plt.ylabel('The Smallest Nonzero Eigenvalue  (λ_2)')
    plt.title('Uniform Hyper-network')
    plt.legend()
    # plt.grid()
    plt.show()

# 主函数
def main():
    filenames = generate_filenames()
    for filename in filenames:
        col1, col2, col3, col4, col5 = process_file(filename)
        if col2 is not None:
            unique_vals, avg_col3, avg_col4, avg_col5 = calculate_averages(col2, col3, col4, col5)
            plot_data(unique_vals, avg_col3, avg_col4, avg_col5)

if __name__ == "__main__":
    main()
