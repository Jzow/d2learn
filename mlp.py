# 多层感知机
import tensorflow as tf
import matplotlib.pyplot as plt


# relu函数，大多数relu函数是非线性函数，非线性意味着函数无法满足加分性质和齐次性质
# 加法性质，给定一个函数，即对于任意实数x和y f(x + y) = f(x) + f(y)
# 齐次性质，对于任意实现x和常数a f(ax) = af(x)
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)

plt.figure(figsize=(5, 2.5))
plt.plot(x.numpy(), y.numpy())
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.title('ReLU Function')
plt.grid()
plt.show()

with tf.GradientTape() as t:
    y = tf.nn.relu(x)

plt.plot(x.numpy(), t.gradient(y, x).numpy())
plt.xlabel('x')
plt.ylabel('grad of relu')
plt.grid()
plt.show()

# sigmoid函数 广泛应用于输出单元上的激活函数，在隐藏层中少见大多数被Relu取代
y = tf.nn.sigmoid(x)
plt.plot(x.numpy(), y.numpy())
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid()
plt.show()

# 计算梯度 求导
with tf.GradientTape as t:
    y = tf.nn.sigmoid(x)

plt.plot(x.numpy(), t.gradient(y, x).numpy())
plt.figure(figsize=(5, 2.5))
plt.xlabel('x')
plt.ylabel('grad of sigmoid')
plt.grid()
plt.show()

