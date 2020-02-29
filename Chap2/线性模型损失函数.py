#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='/Library/Fonts/Songti.ttc')

plt.figure(figsize=(8, 5))

x = np.arange(-10, 10, 0.01)
# 逻辑回归损失函数
logi = np.log(1 + np.exp(-x))
# 感知机损失函数
y_p = -x
y_p[y_p < 0] = 0
# 线性支持向量机
y_hinge = 1.0 - x
y_hinge[y_hinge < 0] = 0

plt.xlim([-3, 3])
plt.ylim([0, 4])
plt.plot(x, logi, 'r-', mec='k', label='Logistic Loss', lw=2)
plt.plot(x, y_p, 'g-', mec='k', label='Perceptron Loss', lw=2)
plt.plot(x, y_hinge, 'b-', mec='k', label='Hinge Loss', lw=2)
plt.grid(True, ls='--')
plt.legend(loc='upper right')
plt.title('各种不同的损失函数', fontproperties=font)
plt.xlabel('yf(x,w)')
plt.ylabel('loss')
plt.show()
