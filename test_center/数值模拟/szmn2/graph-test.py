import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体和字号
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18  # 设置全局字号为18号

# 定义文件前缀和后缀
prefix = ""
suffix = "_5_10_old_1500-error_history.npy"

# 读取文件名范围
start_idx = 10
end_idx = 50

# 自定义图例内容
labels = [f"M={i}" for i in range(start_idx, end_idx + 10, 10)]

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 定义颜色和样式
colors = ['r', 'g', 'b', 'c', 'm']
line_style = '--'

# 循环读取文件并绘制误差曲线
for i in range(start_idx, end_idx + 10, 10):
    filename = f"{prefix}{i}{suffix}"
    try:
        error_history = np.load(filename)
        num_steps = error_history.shape[0]
        ax.plot(
            np.arange(num_steps), 
            error_history, 
            line_style, 
            label=labels[int((i-start_idx)/10)],  # 使用自定义图例内容
            color=colors[int((i-start_idx)/10)]
        )
    except FileNotFoundError:
        print(f"File {filename} not found. Skipping...")

# 添加放大图
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
zoom_ax = inset_axes(ax, width="50%", height="50%", loc=1, borderpad=1)  # 通过数值调整位置和边距

# 在放大图上绘制曲线
for i in range(start_idx, end_idx + 1):
    filename = f"{prefix}{i}{suffix}"
    try:
        error_history = np.load(filename)
        num_steps = error_history.shape[0]
        zoom_ax.plot(
            np.arange(num_steps), 
            error_history, 
            line_style, 
            color=colors[int((i-start_idx)/10)]
        )
    except FileNotFoundError:
        continue

# 去掉小图的横纵坐标标签，但保留刻度线
zoom_ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)  # x轴：只显示刻度线，不显示标签
zoom_ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=False)   # y轴：只显示刻度线，不显示标签

zoom_ax.tick_params(direction='in', length=4, width=1)  # 设置刻度线样式，向内突出

# 设置主图例和标签
ax.set_xlabel('t', fontsize=18, fontstyle='italic')      # 设置为斜体
ax.set_ylabel('E(t)', fontsize=18, fontstyle='italic')   # 设置为斜体
ax.set_title('ER non-uniform hypernetwork', fontsize=18)
ax.legend(loc=(0.15, 0.64), fontsize=14, frameon=True)  # 通过数值调整图例位置
# ax.grid(True)
ax.set_xlim()
ax.set_ylim(0, 1.0)

# 设置放大图范围
zoom_ax.set_xlim(100000, 500000)
zoom_ax.set_ylim(1e-6, 1e-4)
# zoom_ax.set_xlabel('Time Step', fontsize=10)
# zoom_ax.set_ylabel('Error', fontsize=10)
# zoom_ax.grid(True)

# 调整布局并显示图形
# 保存并显示
plt.tight_layout()
plt.savefig("graph_test.svg", dpi=300)
plt.savefig("graph_test.png", dpi=300)
plt.show()