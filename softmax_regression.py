import tensorflow as tf
from matplotlib_inline import backend_inline
from mnist import load_data_mnist, get_mnist_label, show_images
import matplotlib.pyplot as plt

batch_size = 256
train_iter, test_iter = load_data_mnist(batch_size)

# 初始化模型参数
num_inputs = 784
num_outputs = 10

W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))

# 定义softmax
X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(tf.reduce_sum(X, 0, keepdims=True))
print(tf.reduce_sum(X, 1, keepdims=True))

def softmax(X):
    X_exp = tf.exp(X)
    partition = tf.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition

X = tf.random.normal((2, 5), 0, 1)
X_prob = softmax(X)

print("X_prob: ", X_prob)
print("reduce_sum_X_prob", tf.reduce_sum(X_prob, 1))

# 定义模型
def net(X):
    return softmax(tf.matmul(tf.reshape(X, (-1, W.shape[0])), W) + b)

# 定义损失函数
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))

def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

print("cross_entropy: ", cross_entropy(y_hat, y))

# 分类精度
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = tf.argmax(y_hat, axis=1)

    cmp = tf.cast(y_hat, y.dtype) == y
    return float(tf.reduce_sum(tf.cast(cmp, y.dtype)))

print("accuracy: ", accuracy(y_hat, y) / len(y))

# 评估任意模型net的精度
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), tf.size(y))
    return metric[0] / metric[1]

print("evaluate_accuracy: ", evaluate_accuracy(net, test_iter))


print("=========================模型训练==============================")

def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def sgd(params, grads, lr, batch_size):
    for params, grads in zip(params, grads):
        params.assign_sub(lr * grads / batch_size)

def train_epoch_ch3(net, train_iter, loss, updater):
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度
        with tf.GradientTape() as tape:
            y_hat = net(X)
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # keras的loss默认返回一个批量平均损失
        l_sum = l * float(tf.size(y)) if isinstance(loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))

    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * nrows == 1:
            self.axes = [self.axes, ]
            # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_metrics[0]:.4f}, "
              f"Train Acc: {train_metrics[1]:.4f}, "
              f"Test Acc: {test_acc:.4f}")

        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    plt.show()

class Updater():
    """小批量随机梯度下降法更新参数"""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        sgd(self.params, grads, self.lr, batch_size)

updater = Updater([W, b], lr=0.1)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

print("=========================模型推理==============================")

def predict_ch3(net, test_iter, n=6):
    """预测"""
    for X, y in test_iter:
        break
    trues = get_mnist_label(y)
    preds = get_mnist_label(tf.argmax(net(X), axis=1))

    images = tf.reshape(X[0:n], (n, 28, 28))
    fig, axes = plt.subplots(1, n, figsize=(15, 15))

    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.set_title(f'true label: {trues[i]}\npredict label: {preds[i]}')
        ax.axis('off')

    plt.show()

    # titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    # show_images(tf.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])
    # plt.show()

predict_ch3(net, test_iter)


