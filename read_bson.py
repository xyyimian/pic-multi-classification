# This program should be run in Python 3 environment.
# Using following command to install necessary libs if you find libs lack.
#
# $ sudo apt-get install python3-pip
# $ sudo pip3 install numpy
# $ sudo pip3 install pandas
# $ sudo pip3 install matplotlib
# $ sudo pip3 install sklearn
# $ sudo pip3 install scikit-image
# $ sudo pip3 install tensorflow			The lib we get from http://www.tensorfly.cn/ only be supported in Python 2.7, we need to get the version which is supported in python 3.
# $ sudo pip3 install pymongo				We should get the pymongo lib but not the bson lib.DON'T USE COMMAND $ sudo pip3 install bson !


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
    plt.imshow(picture)
    pylab.show()
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
print(Y_flat)

#redimension X for our model
X_flat = X_images.reshape(X_images.shape[0], -1)	


