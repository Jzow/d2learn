# 使用Tensorflow API 实现softmax回归
import tensorflow as tf
from mnist import load_data_mnist
from softmax_regression import train_ch3

batch_size = 256
train_iter, test_iter = load_data_mnist(batch_size)

net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer= weight_init, activation='linear'))

# from_logits=True 最后一层没有应用 softmax 函数
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

trainer = tf.keras.optimizers.SGD(learning_rate=0.1)


# 训练模型
num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)