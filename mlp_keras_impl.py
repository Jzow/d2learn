# 通过使用Tensorflow Keras API实现多层感知器（MLP）

import tensorflow as tf
import matplotlib.pyplot as plt
import common as cm

num_outputs = 10
num_hiddens = 256

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_hiddens, activation='relu'),
    tf.keras.layers.Dense(num_outputs)
])

batch_size, lr, num_epochs = 256, 0.1, 10
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)

train_iter, test_iter = cm.load_data_mnist(batch_size)
cm.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


cm.predict_ch3(net, test_iter)
plt.show()
