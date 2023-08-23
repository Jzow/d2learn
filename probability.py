import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline

fair_probs = tf.ones(6) / 6
print(fair_probs)

# 多项式分布由probs参数化，probs是一个批次
counts = tfp.distributions.Multinomial(1000, fair_probs).sample()
print(counts / 1000)


# 抽取10个样本 进行500组实验
sample = tfp.distributions.Multinomial(10, fair_probs).sample(1000)
cum_counts = tf.cumsum(sample, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)


def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(6, 4.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

set_figsize((6, 4.5))
for i in range(6):
    plt.plot(estimates[:, i].numpy(), label = ("P(die = " + str(i + 1) + ")"))

plt.axhline(y=0.167, color='black', linestyle='dashed')
plt.gca().set_xlabel("x")
plt.gca().set_ylabel("y")
plt.legend();
plt.show();

