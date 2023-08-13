import tensorflow as tf

x = tf.range(12, dtype=tf.float32)
size = tf.size(x)

print(x)
print(size)

X = tf.reshape(x, (3, -1))

zeros = tf.zeros((3, 3, 4))

ones = tf.ones((3, 3, 4))

random = tf.random.normal(shape=[3, 4])
print(random)
print(tf.concat(zeros, axis=0))

print(zeros == ones)

print(tf.reduce_sum(ones))

a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1,2))

print(a)
print(b)

print(a + b)
print(a * b)

## index
print(X)
print(X[-1])
print(X[1: 3])

# TensorFlow中的Tensors是不可变的，也不能被赋值
# TensorFlow中的Variables是支持赋值的可变容器
# TensorFlow中的梯度不会通过Variable反向传播

X_var = tf.Variable(X)
print(X_var[0:2, :].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12))
