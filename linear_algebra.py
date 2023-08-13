import tensorflow as tf

# 标量 scalar
scalar_x = tf.constant(3.0)
scalar_y = tf.constant(2.0)

print("scalar_x+scalar_y: ", scalar_x + scalar_y)
print("scalar_x*scalar_y: ", scalar_x * scalar_y)
print("scalar_x/scalar_y: ", scalar_x / scalar_y)
print("scalar_x**scalar_y: ", scalar_x ** scalar_y)
print("==========================================================")

# 向量 vector
vector_x = tf.range(4)
print("vector_x: ", vector_x)
print("vector_x[3]: ", vector_x[3])
print("vector_x size: ", len(vector_x))
# 如果向量只有一个轴也可以通过shape属性获取size
print("vector_x size: ", vector_x.shape)
print("==========================================================")

# 矩阵 matrix
matrix_A = tf.reshape(tf.range(12), (3, 4))
print("matrix_A: ", matrix_A)
# 通过索引index访问矩阵元素
print("matrix_A[11]: ", matrix_A[2][3])
# 矩阵转置 matrix transpose 交换矩阵的行和列
matrix_A_T = tf.transpose(matrix_A)
print("matrix_A_T: ", matrix_A_T)

# 对称矩阵 即转置后跟转置前元素对齐
matrix_B = tf.constant([[5, 6, 7], [6, 7, 9], [7, 9, 11]])
matrix_B_T = tf.transpose(matrix_B)
print("matrix_B: ", matrix_B)
print("matrix_B == matrix_B_T: ", matrix_B == matrix_B_T)
print("==========================================================")

# 张量 tensor
# 比如 向量是标量的推广，矩阵是向量的推广一样，张量可以构建多个数量轴的n维数组
# 向量是一阶张量，而矩阵是二阶张量
tensor_A = tf.reshape(tf.range(24), (2, 3, 4))
print("tensor_A: ", tensor_A)
print("tensor_A size: ", len(tensor_A))
# 张量加法运算
tensor_B = tf.reshape(tf.range(16, dtype=tf.float32), (4, 4))
tensor_C = tensor_B
print("tensor_B: ", tensor_B)
print("tensor_B + tensor_C: ", tensor_B + tensor_C)
print("tensor_B * tensor_C: ", tensor_B * tensor_C)
# 张量加或乘一个标量不会改变原来的形状，该张量的每个item都会与标量相加或相乘
scalar_z = 2.0
print("tensor_B + scalar_z: ", tensor_B + scalar_z)
print("tensor_B * scalar_z: ", (scalar_z * tensor_B).shape)
print("==========================================================")

# 降维
# 可以对任意一个张量进行元素和
vector_y = tf.range(4, dtype=tf.float32)
print("vector_y sum: ", tf.reduce_sum(vector_y))
print("tensor_A sum: ", tf.reduce_sum(tensor_A))

# 可以指定张量沿着哪一轴通过求和降低维度，这里以矩阵A为例，通过求和所有行的元素来降维（轴0）指定axis=0
matrix_A_sum_axios0 = tf.reduce_sum(matrix_A, axis=0)
print("matrix_A A_sum_axios0: ", matrix_A_sum_axios0)
tensor_A_sum_axios0 = tf.reduce_sum(tensor_A, axis=0)
print("tensor_A A_sum_axios0: ", tensor_A_sum_axios0)
# 同时也可以指定axis=1将通过汇总所有列的元素降维（轴1）
matrix_A_sum_axios1 = tf.reduce_sum(matrix_A, axis=1)
print("matrix_A A_sum_axios1: ", matrix_A_sum_axios1)
tensor_A_sum_axios1 = tf.reduce_sum(tensor_A, axis=1)
print("tensor_A A_sum_axios1: ", tensor_A_sum_axios1)

# 沿着行和列的矩阵或者张量求和，等价于对矩阵或张量所有元素进行求和
matrix_A_sum = tf.reduce_sum(matrix_A, axis=[0, 1])
tensor_A_sum = tf.reduce_sum(tensor_A, axis=[0, 1])
print("matrix_A_sum: ", matrix_A_sum)
print("tensor_A_sum: ", tensor_A_sum)
# 计算矩阵或者张量的平均值，平均值 = 总和 / 元素个数
matrix_A_sum_mean = tf.reduce_mean(matrix_A)
matrix_A_sum_mean_other = tf.reduce_sum(matrix_A) / tf.size(matrix_A).numpy()
print("matrix_A_sum_mean: ", matrix_A_sum_mean)
print("matrix_A_sum_mean_other: ", matrix_A_sum_mean_other)

# 同理可以按照某个维度进行计算平均值
tensor_A_sum_axios0_mean = tf.reduce_mean(tensor_A, axis=0)
print("tensor_A_sum_axios0_mean: ", tensor_A_sum_axios0_mean)
print("==========================================================")

# 非降维求和 keepdims=True
tensor_A_keepdims = tf.reduce_sum(tensor_A, axis=0, keepdims=True)
print("tensor_A_keepdims: ", tensor_A_keepdims)
# 可以通过广播将张量A除以非降维张量A。
print("tensor_A / tensor_A_keepdims: ", tensor_A / tensor_A_keepdims)
# 如果想沿某个轴计算tensor_A元素的累积总和， 比如axis=0（按行计算），可以调用cumsum函数。 此函数不会沿任何轴降低输入张量的维度。
tensor_A_cumsum = tf.cumsum(tensor_A, axis=0)
print("tensor_A_cumsum: ", tensor_A_cumsum)
print("==========================================================")

# 点积 Dot Product
vector_a = tf.range(4, dtype= tf.float32)
vector_b = tf.range(4, dtype= tf.float32)
print("a_b_tensordot: ", tf.tensordot(vector_a, vector_b, axes=1))
# 等同于
tf.reduce_sum(vector_a * vector_b)
print("==========================================================")

# 矩阵 向量积
matrix_C = tf.constant([[5, 6, 7], [6, 7, 9], [7, 9, 11]])
vector_d = tf.range(3)
print("matrix_C: ", matrix_C)
print("vector_d: ", vector_d)
print("matrix_C && vector_d matvec: ", tf.linalg.matvec(matrix_C, vector_d))

matrix_D = tf.ones((3, 3), dtype=tf.int32)
print("matrix_C && matrix_D matmul: ", tf.matmul(matrix_C, matrix_D))
print("==========================================================")


# 范数 norm
# L2范数是向量元素平方和的平方根
vector_number = tf.constant([3, 4], dtype=tf.float32)
print("vector_norm: ", tf.norm(vector_number))
# L1范数是向量元素的绝对值之和
print("vector_abs: ", tf.reduce_sum(tf.abs(vector_number)))
# Frobenius范数是矩阵元素平方和的平方根
print("matrix_one_norm: ", tf.norm(tf.ones((4, 9))))