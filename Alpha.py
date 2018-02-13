import pylab						#This lib is used to show the picture.If you only use `plt.imshow(picture)` python3 won't show the picture
import warnings
import io, bson
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.data import imread   # or, whatever image library you prefer
from sklearn import preprocessing
from subprocess import check_output
import random
import os

#Output the files in the catagory
print(check_output(["ls", "./input"]).decode("utf8"))
#Ignore Warnings
warnings.filterwarnings("ignore")

# Simple data processing
data = bson.decode_file_iter(open('./input/train_example.bson', 'rb'))
# read bson file into pandas DataFrame
with open('./input/train_example.bson','rb') as b:
    df = pd.DataFrame(bson.decode_all(b.read()))

#Get shape of first image
for e, pic in enumerate(df['imgs'][0]):
        picture = imread(io.BytesIO(pic['picture']))
        pix_x,pix_y,rgb = picture.shape

n = len(df.index) #cols of data in train set
X_ids = np.zeros((n,1)).astype(int)
Y = np.zeros((n,1)).astype(int) #category_id for each row
X_images = np.zeros((n,pix_x,pix_y,rgb)) #m images are 180 by 180 by 3

i = 0
for c, d in enumerate(data):
    X_ids[i] = d['_id']
    Y[i] = d['category_id']
    for e, pic in enumerate(d['imgs']):
        picture = imread(io.BytesIO(pic['picture']))
    X_images[i] = picture #add only the last image
    #plt.imshow(x)
    #pylab.show()
    i+=1

#full list of classes
df_categories = pd.read_csv('./input/category_names.csv', index_col='category_id')
category_classes = df_categories.index.values
category_classes = category_classes.reshape(category_classes.shape[0],1)

#using a label encoder, and binarizer to convert all unique category_ids to have a column for each class
le = preprocessing.LabelEncoder()
lb = preprocessing.LabelBinarizer()

le.fit(df_categories.index.values)
y_encoded = le.transform(Y)

lb.fit(y_encoded)
Y_flat = lb.transform(y_encoded)

print ('Input Success\n')

ckpt = tf.train.get_checkpoint_state('./AlexNet')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')

#check variable names
'''
from tensorflow.python import pywrap_tensorflow
reader = pywrap_tensorflow.NewCheckpointReader('./AlexNet/AlexNet.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
'''

print ('Restore Success\n')

#the following example show the result with one picture

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver.restore(sess,ckpt.model_checkpoint_path)
    img_raw = tf.gfile.FastGFile('./try.jpeg', 'rb').read()
    img = sess.run(tf.image.resize_images( tf.image.decode_jpeg(img_raw), [227,227], method=random.randint(0, 3)))
    imgs = []
    for i in range(128):        #duplicate or there would be too little input
        imgs.append(img)
    #print(imgs)
    print(sess.run(tf.get_default_graph().get_tensor_by_name('fc8/weights:0'),feed_dict={'Placeholder:0': [img]}))

'''
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver.restore(sess,ckpt.model_checkpoint_path)
    img = sess.run(tf.image.resize_images(X_images,[227,227],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR ))
    #this kind of preprocess is quite unbearable!!!

    #using all layers expcept the last layer
    
'''

#using all layers except the last one

