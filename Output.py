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



def transfer_data():    # A generator used to generating a batch of labels and images
    #transfer all images to features
    data = bson.decode_file_iter(open(PIC_PATH, 'rb')) # bson.decode_file_iter is a generator
    # full list of classes
    df_categories = pd.read_csv(LABEL_PATH, index_col='category_id')

    # using just binarizer without endcoder to convert all unique category_ids to have a column for each class
    lb = preprocessing.LabelEncoder()
    lb.fit(df_categories.index.values)

    with open("ids.txt","w") as f:
        for c, d in enumerate(data):
            temp = []
            temp.append(d['category_id'])
            zheli = lb.transform(temp)
            f.write(str(zheli[0]) + '\n')


if __name__ == "__main__":
    transfer_data()