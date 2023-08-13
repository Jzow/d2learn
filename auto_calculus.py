import tensorflow as tf

x = tf.range(4, dtype=tf.float32)
print("向量x: ", x)

# 存储梯度，不会在每一次参数求导分配内存
x = tf.Variable(x)

# GradientTape() 梯度流 记录磁带
with tf.GradientTape() as t:
    # 监视watch要求导的变量y
    y = 2 * tf.tensordot(x, x, axes=1)

print("一阶段求导: ", y)

x_grad = t.gradient(y, x);
print("二阶段：", x_grad)
# 得出 函数 y = 2xTx 关于 x 的梯度应该为4，验证一下
print(x_grad == 4 * x)

# 计算x的另一个函数
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)

# 被新计算的梯度覆盖
x_grad_two = t.gradient(y, x)
print(x_grad_two)


# 非标量变量的反向传播
# 这里目的单独计算批量中每个样本的偏导数之和，而并非是计算微分矩阵
# 因为有时候我们需要计算一批样本中每个组成部分的损失函数的导数
with tf.GradientTape() as t:
    y = x * x
print(t.gradient(y, x))


# 分离计算
# 将一些计算移动到记录的计算图之外，假设y是作为x的函数计算的，而z是作为y和x的函数计算的
# 计算z关于x的梯度，突然有原因想把y作为一个常数，只考虑x在被y计算后的结果
# 这种做法可以把y分离出来一个新变量比如u，该变量和y具有相同的值，但丢弃了计算图中如何计算y的信息。

# persistent=True的意思是来运行多次t.gradient()
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad_three = t.gradient(z, x)
x_grad_three == u
print("x_grad_three: ", x_grad_three)
# 验证计算结果
print(t.gradient(y, x) == 2 * x)

# 控制流梯度计算
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c

a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)

d_grad = t.gradient(d, a)
print("d_grad:", d_grad)

# f(a) = k*a k取决于输入的值a 可以验证 d/a梯度是否正确
print(d_grad == d / a)