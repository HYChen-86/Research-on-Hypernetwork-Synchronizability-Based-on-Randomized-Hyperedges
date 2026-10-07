import numpy as np
import matplotlib.pyplot as plt

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
zoom_ax = inset_axes(ax, width="50%", height="50%", loc=1, borderpad=1.5)  # 通过数值调整位置和边距

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

# 设置主图例和标签
legend_fontsize = 10  # 图例字体大小
ax.set_xlabel('t', fontsize=12)
ax.set_ylabel('E(t)', fontsize=12)
ax.set_title('ER non-uniform hypernetwork', fontsize=14)
ax.legend(loc=(0.15, 0.73), fontsize=legend_fontsize, frameon=True)  # 通过数值调整图例位置
# ax.grid(True)
ax.set_xlim(None ,500000)
ax.set_ylim(0, 1.0)

# 设置放大图范围
zoom_ax.set_xlim(10000, 500000)
zoom_ax.set_ylim(1e-6, 5e-3)
# zoom_ax.set_xlabel('Time Step', fontsize=10)
# zoom_ax.set_ylabel('Error', fontsize=10)
# zoom_ax.grid(True)

# 调整布局并显示图形
# 保存并显示
plt.tight_layout()
plt.savefig("graph_test.svg", dpi=300)
plt.savefig("graph_test.png", dpi=300)
plt.show()