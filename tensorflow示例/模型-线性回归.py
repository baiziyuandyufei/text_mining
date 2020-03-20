# coding:utf-8
"""
示例用途：
（1）演示TensorFlow开发的基本步骤
（2）输入节点的定义方法
（3）网络参数的定义方法
（3）正向传播 
（4）反向传播 损失函数 优化函数
（4）网络参数初始化
（5）训练模型
（6）测试模型
（7）图形显示测试模型细节
（8）使用模型
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
#显示模拟数据点
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

# 绘制损失曲线用
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

# 定义输入节点
# 方法1：占位符（一般使用这种方式）
X = tf.placeholder("float")
Y = tf.placeholder("float")
# # 方法2：字典类型（输入比较多时使用）
# inputdict = {
#     'x': tf.placeholder("float"),
#     'y': tf.placeholder("float")
# }

# 定义参数
# # 方法1：模型参数 直接定义模型参数
# W = tf.Variable(tf.random_normal([1]), name="weight")
# b = tf.Variable(tf.zeros([1]), name="bias")
# 方法2：模型参数 字典定义模型参数（常用）
paradict = {
    'w': tf.Variable(tf.random_normal([1])),
    'b': tf.Variable(tf.zeros([1]))
}

# 定义正向传播
z = tf.multiply(X, paradict['w'])+ paradict['b']
# 反向传播 定义损失函数
cost =tf.reduce_mean( tf.square(Y - z))
# 方向传播 学习率
learning_rate = 0.01
# 反向传播 定义优化函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化变量
init = tf.global_variables_initializer()
# 记录训练细节
plotdata = { "batchsize":[], "loss":[] }
# 训练参数
training_epochs = 10
display_step = 2
# 启动session
with tf.Session() as sess:
    # 必须在所有变量和操作的定义完成之后
    sess.run(init)

    # 训练模型
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # 测试模型
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", epoch+1, 
                    "cost=", loss,
                       "W=", sess.run(paradict['w']), 
                       "b=", sess.run(paradict['b']))
            if not (loss == "NA" ):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print (" Finished!")
    print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), 
              "W=", sess.run(paradict['w']), 
              "b=", sess.run(paradict['b']))

    # 图形显示测试模型细节
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(paradict['w']) * train_X + sess.run(paradict['b']), label='Fitted line')
    plt.legend()
    plt.show()

    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')

    plt.show()

    # 使用模型
    print ("x=0.2，z=", sess.run(z, feed_dict={X: 0.2}))