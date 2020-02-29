import tensorflow as tf
import numpy as np

# 定义添加网络层操作
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

# 定义输入层
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
# 定义隐藏层
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# 定义输出层
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)
# 定义损失函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

# 定义训练步骤
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 构建数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
print('x_data.shape', x_data.shape)
print('y_data.shape', y_data.shape)


sess = tf.Session()
# 定义好Session后才能合并所有summary
merged = tf.summary.merge_all()
# 定义好Session后才能定义FileWriter
writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(tf.initialize_all_variables())

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i%50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
    writer.add_summary(result, i)


sess.close()





