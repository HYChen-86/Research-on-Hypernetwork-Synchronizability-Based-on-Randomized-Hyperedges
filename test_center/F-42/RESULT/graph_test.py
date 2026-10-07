import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# 数据读取
def process_file(filepath):
    try:
        data = np.loadtxt(filepath, delimiter=',')  # 假设文件以逗号分隔
        x_values = data[:, 2]  # 第三列数据为 x
        y_values = data[:, 1]  # 第二列数据为 y
        return x_values, y_values
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None

# 数据清理
def clean_data(x, y):
    x = np.array(x)
    y = np.array(y)
    valid_indices = np.isfinite(x) & np.isfinite(y)
    return x[valid_indices], y[valid_indices]

# 主程序
def main():
    # 获取数据
    filenames = []
    for i in range(10, 62, 4):  # 生成文件名
        filenames.append(f"40_5_{i}.txt")
    
    all_x = []
    all_y = []

    for filename in filenames:
        x_values, y_values = process_file(filename)
        if x_values is not None and y_values is not None:
            all_x.extend(x_values)
            all_y.extend(y_values)

    # 转为 numpy 数组并清理数据
    all_x, all_y = clean_data(all_x, all_y)

    # 样条插值拟合
    spline = UnivariateSpline(all_x, all_y)
    spline.set_smoothing_factor(1)  # 调节平滑度

    # 生成拟合曲线
    x_fit = np.linspace(min(all_x), max(all_x), 500)
    y_fit = spline(x_fit)

    # 绘制图像
    plt.figure(figsize=(10, 8))
    plt.scatter(all_x, all_y, alpha=0.7, edgecolors='w', s=60, label="Data Points")
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label="Fitted Curve (Spline)")
    plt.title("Scatter Plot with Cubic Spline Fitting", fontsize=14)
    plt.xlabel("X-axis", fontsize=12)
    plt.ylabel("Y-axis", fontsize=12)
    plt.legend()
    plt.tight_layout()

    # 显示图像
    plt.savefig("test.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
