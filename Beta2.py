import io, bson
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from skimage.data import imread   # or, whatever image library you prefer
from sklearn import preprocessing
import random
import os
from CTF.kaffe.tensorflow import Network
import pickle
import matplotlib.pyplot as plt
import pylab


PIC_PATH = './input/train.bson'
LABEL_PATH = './input/category_names.csv'
#FEATURE_PATH = './input/feature.npy'
MODEL_PATH = './AlexNet'
STORED_PATH = './model/AlexNet'
BATCH_SIZE = 500   # There are 12371293 images in total, a little less or more
CAT_NUM = 5270
LEARNING_RATE = 0.001
MOMENTUM = 0.9
MODEL_SIZE = 227
RANGE = 24742


class PreTrainedNet(Network):
    #little network utilizing pretrained model results as features
    def setup(self):
        (self.feed('data')
             .fc(CAT_NUM, name='ip1')
             .softmax(name='prob'))




def generating_feature():   # Generating feature file
    ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
    print(tf.global_variables())
    #f = open(FEATURE_PATH, 'wb')
    resized_data = transfer_data()
    config = tf.ConfigProto()  
    config.gpu_options.per_process_gpu_memory_fraction = 0.7   # Restrict the growth of memory use, or memory will be used up

    result_op = tf.get_default_graph().get_tensor_by_name("fc7/fc7:0")

    with tf.Session(config = config) as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        #weight_op = tf.get_default_graph().get_tensor_by_name("fc7/weights:0")
        #bias_op = tf.get_default_graph().get_tensor_by_name("fc7/biases:0")
        #result_op = tf.get_default_graph().get_tensor_by_name("fc7/fc7:0")
        i = 0
        for X_flat, Y_flat in resized_data:
            i += 1
            if i == RANGE:                                                  # Some errors in using iterator transfer_data()
                resized_data.close()                                        # So it's necessary to close it manually.
            result = sess.run(result_op, feed_dict = {'Placeholder:0': X_flat})
            yield result, Y_flat







def transfer_data():    # A generator used to generating a batch of labels and images
    #transfer all images to features
    data = bson.decode_file_iter(open(PIC_PATH, 'rb')) # bson.decode_file_iter is a generator
    # full list of classes
    length = len(data)
    print length
    char = input()
    df_categories = pd.read_csv(LABEL_PATH, index_col='category_id')
    category_classes = df_categories.index.values
    category_classes = category_classes.reshape(category_classes.shape[0], 1)

    # using just binarizer without endcoder to convert all unique category_ids to have a column for each class
    lb = preprocessing.LabelBinarizer()
    lb.fit(df_categories.index.values)

    # Size of pictures is defined here instead read size of the first picture
    n = 500 # Batch size
    pix_x = 180
    pix_y = 180
    rgb = 3

    X_ids = np.zeros((n, 1)).astype(int)
    Y = np.zeros((n, 1)).astype(int)  # category_id for each row
    X_images = np.zeros((n, pix_x, pix_y, rgb))  # m images are 180 by 180 by 3
    i = 0

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7   # Restrict the growth of memory use, or memory will be used up
    images = tf.placeholder(tf.float32, [n, pix_x, pix_y, rgb], name = "images")
    op = tf.image.resize_images(images, [MODEL_SIZE, MODEL_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    batch_num = 0

    with tf.Session(config = config) as sess:
        for c, d in enumerate(data):
            for e, pic in enumerate(d['imgs']):
                if i == 0:
                    Y = np.zeros((n, 1)).astype(int)  # category_id for each row
                    X_images = np.zeros((n, pix_x, pix_y, rgb))  # m images are 180 by 180 by 3
                picture = imread(io.BytesIO(pic['picture'])) # All images should be added.
                Y[i] = d['category_id']
                X_images[i] = picture
                i += 1
                if i == n:
                    batch_num += 1
                    i = 0
                    Y_flat = lb.transform(Y)
                    X_flat = sess.run(op, feed_dict = {images: X_images})
                    Y = np.zeros((n, 1)).astype(int)  # category_id for each row
                    X_images = np.zeros((n, pix_x, pix_y, rgb))  # m images are 180 by 180 by 3
                    yield X_flat, Y_flat


#batches generators

# def gen_data(source):
#     data = pickle.load(open(source))
#     while True:
#         indices = range(len(data))
#         random.shuffle(indices)
#         for i in indices:
#             image = data[0][i]
#             label = data[1][i]
#             #print(image, label)
#             yield image, label


# def gen_data_batch(source):
#     data_gen = gen_data(source)
#     while True:
#         image_batch = []
#         label_batch = []
#         for _ in range(BATCH_SIZE):
#             image, label = next(data_gen)
#             image_batch.append(image)
#             label_batch.append(label)
#         yield np.array(image_batch), np.array(label_batch)
#                 #Export X_flat and Y_flat


def training():

    #definitions of operations
    #input layer
    features = tf.placeholder(tf.float32, [BATCH_SIZE, 4096],name="feature")
    #output layer
    labels = tf.placeholder(tf.float32, [BATCH_SIZE, CAT_NUM], name='label')
    net = PreTrainedNet({'data': features})
    ip1 = net.layers['ip1']
    pred = tf.nn.softmax(ip1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ip1, labels = labels), 0)
    #opt = tf.train.MomentumOptimizer(learning_rate = LEARNING_RATE, momentum = MOMENTUM)
    opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train_op = opt.minimize(loss)

    #start training
    config = tf.ConfigProto()  
    config.gpu_options.per_process_gpu_memory_fraction = 0.7

    with tf.Session(config = config) as sesh:
        sesh.run(tf.global_variables_initializer())
        data_gen = generating_feature()
    	#print saver
        i = 0
        for np_features, np_labels in data_gen:
            #print saver
            feed = {features: np_features, labels: np_labels}
            np_loss, np_pred, _ = sesh.run([loss, pred, train_op], feed_dict=feed)
            i += 1
            print('Iteration: ', i, np_loss)
            if i % 100 == 0:
                with tf.variable_scope("ip1", reuse=True):  # Only variables in graph "ip1" needed to be stored
                    saver = tf.train.Saver([tf.get_variable("weights"), tf.get_variable("biases")])
                    saver.save(sesh, STORED_PATH)
                    print('Successfully saved.')
        print('Successful Transforemd.')





if __name__ == "__main__":
    training()
 
