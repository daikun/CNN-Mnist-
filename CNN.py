import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_save_path = './minist_model/'
model_name = 'MODEL'

mnist = input_data.read_data_sets("./data", one_hot=True)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    #成员为True或False
    correct_prediction = tf.equal(tf.argmax(y_pre, 1)
                                  ,tf.argmax(v_ys, 1))
    #tf.cast将True、False转为1、0
    accuracy = tf.reduce_mean(tf.cast(correct_prediction
                                      , tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

#权重，在卷积神经网络中权重即为卷积核
def weight_variable(shape):
    #tf.truncated_normal是谷歌定义的卷积神经网络产生随机数的函数
    initial = tf.truncated_normal(shape= shape,stddev=0.1)
    w = tf.Variable(initial,dtype=tf.float32)
    return w

#偏置量
def get_bias(shape):
    b = tf.Variable(tf.constant(0.01,shape = shape))
    return b

#2*2卷积层。x为输入，W为权重
def conv2d(x,W):
    # tf.nn.conv2d为生成卷积层的默认函数，x为输入,W为权重
    # strides为步长。第一个和第四个规定为1.第2、3个参数为
    # x、y方向步长.
    #padding = 'SAME'指卷积后图片的尺寸和原图片保持一致
    return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = 'SAME')

def max_pooling(x):
    # ksize为池化层尺寸2*2。stride为步长。
    # 虽然padding为SAME。但只有步长为1时，才保持输出图像和输入图像的尺寸相同
    # 步长为2的话，输出图像的尺寸为输入图像的一半。
    return tf.nn.max_pool(x,ksize = [1,2,2,1], strides = [1,2,2,1],
                          padding = 'SAME')

xs = tf.placeholder(tf.float32,shape=(None,784))
ys = tf.placeholder(tf.float32,shape=(None,10))
keep_prob = tf.placeholder(tf.float32)

#将输入的xs转变为图片的模式。28*28*1分别指高、宽、通道
#-1指大小视情况，根据输入自行计算。
x_image = tf.reshape(xs,[-1,28,28,1])

############  START：卷积层1+池化层1  ############
#针对一张图片
#[5,5,1,32]指卷积核为5*5，通道为1，32个卷积核
W_conv1 = weight_variable([5,5,1,8])
#Wx + b。输出的图像为28*28*32。b的数值为输出图像的深度
b_conv1 = get_bias([8])

#第一层卷积层计算relu(Wx + b)  28*28*32
layer1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)

#第一层池化层  #14*14*32
pooling1 = max_pooling(layer1)
############  END：卷积层1+池化层1  ############


############  START：卷积层2+池化层2  ############
#针对一张图片
#[5,5,1,32]指卷积核为5*5，通道为32，64个卷积核
W_conv2 = weight_variable([5,5,8,16])
#Wx + b。输出的图像为28*28*64。b的数值为输出图像的深度
b_conv2 = get_bias([16])

#第二层卷积层计算relu(Wx + b)  28*28*64
layer2 = tf.nn.relu(conv2d(pooling1,W_conv2) + b_conv2)

#第二层池化层  #7*7*64
pooling2 = max_pooling(layer2)
############  END：卷积层2+池化层2  ############


############  START：全连接层1  ############
W_fc1 = weight_variable([7*7*16,100])
b_fc1 = get_bias([100])

#将池化层与全连接层间的图像由三维变为二维
#[n_example,7*7*64,1]。n_example指的图片个数
h_pool2_flat = tf.reshape(pooling2,[-1,7*7*16])
#全连接层1的输出。
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
#h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

############  END：全连接层1  ############


############  START：全连接层2  ############
W_fc2 = weight_variable([100,10])
b_fc2 = get_bias([10])

#全连接层2的输出。
prediction = tf.matmul(h_fc1,W_fc2) + b_fc2

############  END：全连接层2  ############

#prediction:?*10.   tf.argmax(ys,1):?*1
#sparse_softmax_cross_entropy_with_logits的logits为神经网络的输出
#labels为一个数值，指最大值的索引。
#如[0,0.6,0,0]，则labels为1
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=prediction, labels=tf.argmax(ys, 1))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)  # 激活整个神经网络的结构
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(10)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images,mnist.test.labels))















