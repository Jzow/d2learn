# 矢量化计算
# 其目的是利用线性代数库或者深度学习框架进行计算而不是使用语言的for循环
# 加快计算的目的和减少计算开销
import math
import time
import numpy as np
import tensorflow as tf
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt

n = 1000
a = tf.ones([n])
b = tf.ones([n])
print("a :", a)

class Timer:
    '''记录运行多长时间'''
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        '''启动记时器'''
        self.tik = time.time()

    def stop(self):
        '''停止计时器，并将记录保存到记录列表中'''
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        '''返回平均时间'''
        return sum(self.times) / len(self.times)

    def sum(self):
        '''返回时间总和'''
        return sum(self.times)

    def cumsum(self):
        '''返回累计时间'''
        return np.array(self.times).cumsum().tolist()

# 使用for循环看需要多长时间
c = tf.Variable(tf.zeros(n))
timer = Timer()
for i in range(n):
    c[i].assign(a[i] + b[i])

print(f'{timer.stop():.5f} sec')

# 第二种方法更快
timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')

# 正态分布(高斯分布)与平方损失

# 正态分布函数
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

# 使用Numpy进行可视化
x = np.arange(-7, 7, 0.01)
# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]


plt.xlabel('x')
plt.ylabel('p(x)')
plt.figsize=(4.5, 2.5)

for mu, sigma in params:
    plt.plot(x, normal(x, mu, sigma), label=f'mean {mu}, std {sigma}')

plt.legend()
plt.show()