import random
import tensorflow as tf
import matplotlib.pyplot as plt
# 生成1000个样本和数据集
def synthetic_data(w, b, num_examples):
    """生成y = Xw+b+ Noice（噪声）"""
    X = tf.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    # 将函数y的形状调整从(num_examples, 1) 调整为 (-1, 1)
    y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = tf.reshape(y, (-1, 1))
    return X, y

true_w = tf.constant([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0], '\nlabel', labels[0])
plt.scatter(features[:, (1)].numpy(), labels.numpy(), 1)
plt.xlabel('Features')
plt.ylabel('Labels')
plt.title('Synthetic Data')
plt.show()


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机取样本
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 初始化模型参数 trainable=True 表示该张量在训练过程中数据会被更新
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01), trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)

# 定义模型
def linear(X, w, b):
    """线性回归模型"""
    return tf.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2

def sgd(params, grads, lr, batch_size):
    for params, grads in zip(params, grads):
        params.assign_sub(lr * grads / batch_size)

# 训练模型
lr = 0.03
num_epochs = 3
net = linear
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            # X和y的小批量损失
            l = loss(net(X, w, b), y)
        # 计算l关于[w, b]的梯度
        dw, db = g.gradient(l, [w, b])
        # 使用参数的梯度更新参数
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_sum(train_l)):f}')

print(f'w的估计误差: {true_w - tf.reshape(w, true_w.shape)}')
print(f'b的估计误差: {true_b - b}')