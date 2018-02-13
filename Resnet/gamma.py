import io, bson
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from skimage.data import imread   # or, whatever image library you prefer
from sklearn import preprocessing
import random
import os
import sys
sys.path.append("..")
from CTF.kaffe.tensorflow import Network
import pickle
import matplotlib.pyplot as plt
import pylab
from resnet import ResNet50


PIC_PATH = '../input/train.bson'
LABEL_PATH = '../input/category_names.csv'
STORED_PATH = '../back/Resnet'
BATCH_SIZE = 98   # There are 12371293 images in total, a little less or more
CAT_NUM = 5270
LEARNING_RATE = 0.01
MOMENTUM = 0.9
MODEL_SIZE = 224
RANGE = 126237



# global var
n = BATCH_SIZE # Batch size
pix_x = 180
pix_y = 180
rgb = 3


def transfer_data():    # A generator used to generating a batch of labels and images
	#transfer all images to features
	data = bson.decode_file_iter(open(PIC_PATH, 'rb')) # bson.decode_file_iter is a generator

	# full list of classes
	df_categories = pd.read_csv(LABEL_PATH, index_col='category_id')
	category_classes = df_categories.index.values
	category_classes = category_classes.reshape(category_classes.shape[0], 1)

	# using just binarizer without endcoder to convert all unique category_ids to have a column for each class
	lb = preprocessing.LabelBinarizer()
	lb.fit(df_categories.index.values)

	# Size of pictures is defined here instead read size of the first picture
	

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
	



def train():
	
	images = tf.placeholder(tf.float32, [BATCH_SIZE, MODEL_SIZE, MODEL_SIZE, rgb], name = "images")
	labels = tf.placeholder(tf.float32, [BATCH_SIZE, CAT_NUM], name = 'label')
	
	net = ResNet50({'data':images})
	final_layer = net.layers['prob']
	pred  = tf.nn.softmax(final_layer)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = final_layer, labels = labels),0)
	opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
	train_op = opt.minimize(loss)
	
	
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.7
	saver = tf.train.Saver()
	with tf.Session(config = config) as sess:
		sess.run(tf.global_variables_initializer())
		resized_data = transfer_data()
		for i in range(RANGE):
			X_image, Y_flat = next(resized_data)
			#print X_image, Y_flat
			np_loss, np_pred, _ = sess.run([loss, pred, train_op], feed_dict={images: X_image, labels: Y_flat})
			if i % 10 == 0:
				print('Iteration: ', i * 10, np_loss)
			if i % 100 == 0:
				saver.save(sess, STORED_PATH + 'model.ckpt')
	print ("Succ!")
		
if __name__ == "__main__":
	train()
	
	

	
	
		
	
