import tensorflow as tf
from dataset import Dataset
from dataset_InftyCDB_3 import InftyCDB
import matplotlib.pyplot as plt
import numpy as np
import global_config
from tensorflow.examples.tutorials.mnist import input_data

# 初始化权值
def weight_variable(shape):
    # 生成一个截断的正态分布
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

class SymbolClasserCNN():

    # datas = input_data.read_data_sets('MNIST_data', one_hot=True)

    # datas = Dataset(filename="E:/PyCharm/MER_2/data/symbols2.data")

    x = tf.placeholder(tf.float32,[None,global_config.IMG_SIZE*global_config.IMG_SIZE]) #因为图片是100*100，转为一维作为输入层输入
    y = tf.placeholder(tf.float32,[None,global_config.SHAPES_LEN])

    x_image = tf.reshape(x,[-1,global_config.IMG_SIZE,global_config.IMG_SIZE,1])

    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # 28*28的图片第一次卷积后任然为100*100(因为padding格式为SAME，会补零)，第一次池化后变为50*50
    # 第二次卷积后任然为50*50，第二次池化后变为25*25
    # 经过两次卷积及池化后，得到64张25*25的平面

    # 全连接层1
    W_fc1 = weight_variable([25*25*64,1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2,[-1,25*25*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    # 表示神经元输出的概率
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    # 增加一个全连接层2
    W_fc2 = weight_variable([1024, 512])
    b_fc2 = bias_variable([512])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    # 表示神经元输出的概率
    # keep_prob = tf.placeholder(tf.float32)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    W_fc3 = weight_variable([512,global_config.SHAPES_LEN])
    b_fc3 = bias_variable([global_config.SHAPES_LEN])
    prediction = tf.nn.softmax(tf.matmul(h_fc2_drop,W_fc3)+b_fc3)

    # 交叉熵代价函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    # 使用AdamOptimizer优化器
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # argmax返回一维张量中最大的值的位置
    correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    saver = tf.train.Saver()

    batch_size = 100
    test_batch_size = 100
    def __init__(self,datas=None,model=None):
        self.model = model
        self.datas = datas
        if self.datas:
            self.n_batch = self.datas.trainDataLen // self.batch_size
            self.test_n_batch = self.datas.testDataLen // self.test_batch_size

    def train(self,modelSave):
        print("开始训练模型")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if self.model:
                self.saver.restore(sess, self.model)
            for epoch in range(5):
                # print("W_conv1:", sess.run(self.W_conv1))
                self.datas.currTrainLen = 0
                self.datas.currTestLen = 0
                for batch in range(self.n_batch):
                    batch_xs,batch_ys = self.datas.nextTrainBatch(self.batch_size)
                    sess.run(self.train_step,feed_dict={self.x:batch_xs,self.y:batch_ys,self.keep_prob:0.7})

                # 计算准确率
                ave_acc = 0
                for batch in range(self.test_n_batch):
                    batch_xs, batch_ys = self.datas.nextTestBatch(self.test_batch_size)
                    acc = sess.run(self.accuracy,feed_dict={self.x:batch_xs,self.y:batch_ys,self.keep_prob:0.7})
                    ave_acc += acc/self.test_n_batch

                print("Iter:",epoch,"Accuracy:",ave_acc)
                # print("W_conv1:",sess.run(self.W_conv1))
            if modelSave:
                self.saver.save(sess,modelSave)
            elif self.model:
                self.saver.save(sess,self.model)
        print("模型训练结束")

    def predict(self,image):
        print("开始预测")
        if self.model==None:
            print("ERROR:请传入需要的模型")
            return
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess,self.model)
            predict_reses = sess.run(self.prediction,feed_dict={self.x:image,self.keep_prob:0.7})
        Tmin = 0.4
        Tacc = 0.95
        candidata_syms_res = []
        candidata_syms_p_res = []
        for predict_res in predict_reses:
            predict_res_p = np.array(predict_res)/np.sum(predict_res)
            candidata_syms = []
            candidata_syms_p = []
            psum = 0
            while psum<Tacc:
                p_max = np.max(predict_res_p)
                max_idx = np.argmax(predict_res_p)
                if p_max>Tmin:
                    candidata_syms.append(global_config.ALL_SHAPES[max_idx])
                    candidata_syms_p.append(p_max)
                    predict_res_p[max_idx] = 0
                    psum+=p_max
                else:
                    break
            candidata_syms_res.append(candidata_syms)
            candidata_syms_p_res.append(candidata_syms_p)

        print("已返回预测出的候选结果")
        # print("candidata_syms:",candidata_syms_res)
        # print("candidata_syms_p:",candidata_syms_p_res)
        return candidata_syms_res,candidata_syms_p_res

# network/mer_net_InftyCDB_OneWeight_AddLevels3.ckpt效果较好
# network/mer_net_InftyCDB_OneWeight_AddLevels4.ckpt效果更好
# network/mer_net_InftyCDB_OneWeight_AddLevels4_add_i.ckpt效果更好
#network/mer_net_InftyCDB_OneWeight_AddLevels4_add_i_greater.ckpt补上大于等于符号

if __name__ == "__main__":
    # datas = InftyCDB(filename="E:/PyCharm/MER_2/data/symbols4.data")
    # datas.loadData()
    datas = Dataset(filename="E:/PyCharm/MER_2/data/symbols5_4_i.data")
    model = SymbolClasserCNN(datas=datas,model="network/mer_net_InftyCDB_OneWeight_AddLevels5_1.ckpt")
    model.train(modelSave="network/mer_net_InftyCDB_OneWeight_AddLevels5_1.ckpt")
    # datas = Dataset(filename="E:/PyCharm/MER_2/data/number_symbols.data")
    # for i in range(10):
    #     model.predict(model.datas.test.images[i:i+1],"network/mer_net_InftyCDB.ckpt")
    #     print("真实结果为:",np.argmax(model.datas.test.labels[i]))
    #     image = model.datas.test.images[i].reshape(28,28)
    #     plt.imshow(image, cmap="Greys", interpolation="None")
    #     plt.show()