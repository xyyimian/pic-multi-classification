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
from Resnet.network import *
from Resnet.resnet import ResNet50
import matplotlib.pyplot as plt
#import pylab


LEARNING_RATE = 0.01
TEST_PATH = './input/test.bson'
LABEL_PATH = './input/category_names.csv'
MODEL_PATH = './model'
OUTPUT_PATH = './output/output.txt'
BATCH_SIZE = 50   # There are 12371293 images in total, a little less or more
CAT_NUM = 5270
MOMENTUM = 0.9
MODEL_SIZE = 224
RANGE = 35360
#TEST_FEATURE_PATH = './test_feature.npy'
#TRAIN_FEATRUE_PATH = './input/feature.npy'


# class PreTrainedNet(Network):
#     #little network utilizing pretrained model results as features
#     def setup(self):
#         (self.feed('data')
#              .fc(CAT_NUM, name='ip1')
#              .softmax(name='prob'))


# def gen_data(source):
#     data = pickle.load(open(source))
#     while True:
#         indices = range(len(data))
#         random.shuffle(indices)
#         for i in indices:
#             image = data[0][i]
#             label = data[1][i]
#             print image, label
#             yield image, label

# def gen_data_batch(source):
#     data_gen = gen_data(source)
#     while True:
#         image_batch = []
#         label_batch = []
#         for i in range(BATCH_SIZE):
#             image, label = next(data_gen)
#             image_batch.append(image)
#             label_batch.append(label)
#         yield np.array(image_batch), np.array(label_batch)
#                 #Export X_flat and Y_flat            
n = BATCH_SIZE            
pix_x = 180
pix_y = 180
rgb = 3





def transfer_data():    # A generator used to generating a batch of labels and images
	#transfer all images to features
	data = bson.decode_file_iter(open(TEST_PATH, 'rb')) # bson.decode_file_iter is a generator

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
	config.gpu_options.per_process_gpu_memory_fraction = 0.5   # Restrict the growth of memory use, or memory will be used up
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
				Y[i] = d['_id']
				# print(Y[i])
				# sb2000 = input()
				X_images[i] = picture
				i += 1
				if i == n:
					batch_num += 1
					i = 0
					# Y_flat = lb.transform(Y)
					X_flat = sess.run(op, feed_dict = {images: X_images})
					X_images = np.zeros((n, pix_x, pix_y, rgb))  # m images are 180 by 180 by 3
					yield lb, X_flat, Y




# def generating_feature():   # Generating feature file
#     ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
#     saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
	
#     resized_data = transfer_data()
#     config = tf.ConfigProto()  
#     config.gpu_options.per_process_gpu_memory_fraction = 0.7   # Restrict the growth of memory use, or memory will be used up

#     with tf.Session(config = config) as sess:
#         saver.restore(sess, ckpt.model_checkpoint_path)        
#         #weight_op = tf.get_default_graph().get_tensor_by_name("fc7/weights:0")
#         #bias_op = tf.get_default_graph().get_tensor_by_name("fc7/biases:0")
#         result_op = tf.get_default_graph().get_tensor_by_name("fc7/fc7:0")
#         i = 0
#         for X_flat, Y_flat in resized_data:
#             i += 1
#             result = sess.run(result_op, feed_dict = {'Placeholder:0': X_flat})
#             s = repr(i) + ' batch of data has been output'
#             print s
#     f.close()
#     print ('Features generating successfully\n')

	
def test():
	ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
	# saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
	images = tf.placeholder(tf.float32, [BATCH_SIZE, MODEL_SIZE, MODEL_SIZE, rgb], name = "images")
 
	
	
	#start testing
	config = tf.ConfigProto()  
	config.gpu_options.per_process_gpu_memory_fraction = 0.7
	f = open(OUTPUT_PATH, 'wb')

	net = ResNet50({'data':images})
	probs = net.get_output()
	final_layer = net.layers['fc1000']
	pred = tf.nn.softmax(final_layer)

	saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
	# print(tf.all_variables())
	with tf.Session(config = config) as sess:
		data_gen = transfer_data()
		sess.run(tf.initialize_all_variables())
		saver.restore(sess, ckpt.model_checkpoint_path)

		for i in range(RANGE):
			lb, np_features, _id = next(data_gen)				#lb is used to get the real id of goods
			feed = {images: np_features}
			print("########################")	
			print(np_features)
			print("####")
			w123 = sess.run(final_layer,feed_dict=feed)
			# print(sess.run(pred))
			# print(sess.run(tf.get_default_graph().get_tensor_by_name("fc1000/biases:0")))

			np_pred = sess.run(probs,feed_dict=feed)
			print(np_pred)
			print("#########")
			ohou = np.zeros((BATCH_SIZE, 5270), dtype=np.int)
			for sb in xrange(BATCH_SIZE):
				# print _id[sb],np_pred[sb]
				temp = np_pred[sb].tolist()
				ohou[sb][temp.index(max(temp))] = 1
			pred_ids = lb.inverse_transform(ohou)					#Use ids[i] to get the real id
			real_ids = lb.inverse_transform(_id)
			for index in xrange(BATCH_SIZE):
				strtemp = str(_id[index][0])+','+str(pred_ids[index])
				print(strtemp)
				# print(str(_id[index][0]))
				# print(str(pred_ids[index]))
				# str = str(_id[index][0])+","+str(pred_ids[index])
				# print(str)
				# print("#######")
				# print(pred_ids[index])
			# print(_id[])
			# print("##########")
			# print(pred_ids)
			# print tf.argmax(_id),tf.argmax(np_pred)
			# print(_id+"#####"+np_pred)
			
			# f.write([_id,np_pred])
			# # break
			# if i == 2:
			# # 	break;
			if i % 10 == 0:
				print 'Iteration: ', i
		
		#saver = tf.train.Saver()
		#saver.save(sess, STORED_PATH)
		print 'Successful Transforemd.'
	f.close()		 
	
	
if __name__ == "__main__":
	#generating_feature()
	test()

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
