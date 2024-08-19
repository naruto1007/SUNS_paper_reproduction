# from Output_Info_All_cal.mat for every kind of data
# the mat file includes list_Recall, list_Precision, list_F1, list_time,list_time_frame
# a list_F1 includes n scores, n=6 or n=10 maybe

# from list_Recall, list_Precision, list_F1, we can get a mean value and a std value (error bar), n=6 or n=10
# we get 4 methods results with a mean and a std value for every method

# legend: method name
# x: Recall Precision F1
# data: calculated mean value and std, for R, P, F1

import numpy as np
import matplotlib.pyplot as plt

# 假设我们有以下四组数据
recall = [0.8, 0.7, 0.9, 0.6]
precision = [0.7, 0.6, 0.8, 0.5]
f1 = [0.75, 0.65, 0.85, 0.55]
mean_std = [(0.75, 0.05), (0.65, 0.04), (0.85, 0.03), (0.55, 0.02)]

# 设置横坐标标签
x_labels = ['Recall', 'Precision', 'F1']
x = np.arange(len(x_labels))

# 设置纵坐标范围
y_min = min([min(m - s, m + s) for m, s in mean_std])
y_max = max([max(m - s, m + s) for m, s in mean_std])
y_range = y_max - y_min
y_labels = ['SUNS', 'STNeuroNet', 'CaImAn Batch', 'Suite2p']

# 绘制柱状图
fig, ax = plt.subplots()
width = 0.2
for i in range(len(recall)):
    ax.bar(x - width + i * width, [recall[i], precision[i], f1[i]], width, label=y_labels[i])
    ax.errorbar(x - width + i * width, [recall[i], precision[i], f1[i]],
                yerr=[mean_std[i][1], mean_std[i][1], mean_std[i][1]], fmt='o', capsize=5)

# 设置横坐标标签和刻度
fontsize = 14
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=fontsize)
ax.set_ylim(0.1, 1)
plt.yticks([0.1, 0.4, 0.7, 1], fontsize=fontsize - 2)
ax.set_ylabel("Score", fontsize=fontsize + 2)

# 添加图例
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=fontsize - 2, borderaxespad=0.)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(direction='in')
# ax.tick_params(axis='x', direction='in', grid_alpha=0.5)

ax.set_title('ABO 275 μm to 175 μm', fontsize=fontsize)

# 保存或显示图形
plt.savefig("2_cal_F1_step2.png")
plt.show()

