# 多层感知机实现
import tensorflow as tf
import matplotlib.pyplot as plt
import common as cm

batch_size = 256

train_iter, test_iter = cm.load_data_mnist(batch_size)

# 由于minst的图像都是由 28 x 28 = 784 灰色图像像素值组成，有10个图像分类
# 将每个图像看做784个输入特征和10个类别的简单分类数据集
# 实现一个具有隐藏层的的多层感知器，包含了256个隐藏单元

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = tf.Variable(tf.random.normal(shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))
W2 = tf.Variable(tf.random.normal(shape=(num_hiddens, num_hiddens), mean=0.1, stddev=0.01))
b2 = tf.Variable(tf.zeros(num_hiddens))
W3 = tf.Variable(tf.random.normal(shape=(num_hiddens, num_outputs), mean=0.1, stddev=0.01))
b3 = tf.Variable(tf.zeros(num_outputs))


# 激活函数
def relu(X):
    return tf.math.maximum(X, 0)

# 模型
def net(X):
    X = tf.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2

# 损失函数 计算交叉熵损失的函数 评估模型的预测结果(y_hat)与真实标签(y)之间的误差
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(y, y_hat, from_logits=True)

# 训练
num_epochs = 20
lr = 0.1
updater = cm.Updater([W1, W2, b1, b2], lr)
cm.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# 推测
cm.predict_ch3(net, test_iter)
plt.show()