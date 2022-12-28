import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import os
import cv2
from sklearn.model_selection import train_test_split
import warnings
import math

def train_test_split_tensors(X, y, **options):
    """
    encapsulation for the sklearn.model_selection.train_test_split function
    in order to split tensors objects and return tensors as output

    :param X: tensorflow.Tensor object
    :param y: tensorflow.Tensor object
    :dict **options: typical sklearn options are available, such as test_size and train_size
    """
    
    # X = np.array(X)
    # y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), **options)

    X_train, X_test = tf.constant(X_train), tf.constant(X_test)
    y_train, y_test = tf.constant(y_train), tf.constant(y_test)

    return X_train, X_test, y_train, y_test

def import_img(path):
    training_data = []
    for img in os.listdir(path):
        try :
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
            img_array = cv2.resize(img_array, (84, 84))
            img_array = np.squeeze(img_array).astype('float64')
            img_array /= 255
            training_data.append(img_array)
        except:
            pass      
        
    training_data = tf.convert_to_tensor(np.array(training_data))
    
    return training_data

# path = '/home/atik/Documents/UMAML_FSL/data/train/n01532829'

X1 = import_img('/home/atik/Documents/UMAML_FSL/data/train/n01532829')
#y1 = tf.convert_to_tensor(np.zeros(np.array(X1.shape[0]).astype('int32')).astype('float32'))
y1 = tf.convert_to_tensor(np.zeros(len(X1)).astype('float32'))
X2 = import_img('/home/atik/Documents/UMAML_FSL/data/train/n02113712')
y2 = tf.convert_to_tensor(np.zeros(len(X2)).astype('float32')+1)
    
X = tf.concat([X1, X2], axis=0)
y = tf.concat([y1, y2], axis=0)

x_train1, x_test1, y_train1, y_test1 = train_test_split_tensors(X, y, test_size=0.1, shuffle=False)


from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import TimeDistributed, Input

input_shape = (84,84, 3)

def conv_net():
    convnet = Sequential()
    for i in range(4):
        convnet.add(Conv2D(64,(3,3),padding='same',input_shape=input_shape))
        convnet.add(BatchNormalization())
        convnet.add(Activation('relu'))
        convnet.add(MaxPooling2D())
    convnet.add(Flatten())
    return convnet

conv = conv_net()

#conv_5d = TimeDistributed(conv)

# Input samples
sample = Input(input_shape)
sample_feature = conv(sample)

# Input Queries
query = Input(input_shape)
query_feature = conv(query)

def reduce_tensor(x):
    return tf.reduce_mean(x, axis=1)

def reshape_query(x):
    return tf.reduce_mean(x, axis=1)

class_center = Lambda(reduce_tensor)(sample_feature)
query_feature = Lambda(reshape_query)(query_feature)

def prior_dist(x):
    feature, pred = x
    pred_dist = tf.reduce_sum(pred ** 2, axis=1, keepdims=True)
    feature_dist = tf.reduce_sum(feature ** 2, axis=1, keepdims=True)
    dot = tf.matmul(pred, tf.transpose(feature))
    return tf.nn.softmax(-(tf.sqrt(pred_dist + tf.transpose(feature_dist) - 2 * dot)))

def proto_dist(x):
    
    def calc_euclidian_dists(x, y):
        """
        Calculate euclidian distance between two 3D tensors.
        Args:
            x (tf.Tensor):
            y (tf.Tensor):
        Returns (tf.Tensor): 2-dim tensor with distances.
        """
        n = x.shape[0]
        m = y.shape[0]
        #print(n, m)
        # print(tf.expand_dims(x, 1), [1, m, 1])
        # x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
        # print(x)
        # y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
        #print(tf.expand_dims(y, 1))
        x = tf.reduce_mean(tf.expand_dims(x, 1))
        y = tf.reduce_mean(tf.expand_dims(y, 1))
        
        dist = tf.math.pow(x - y, 2)
        #dist = tf.math.multiply(dist,10**5)
        dist = tf.expand_dims(dist,0)
        #print(dist)
        return dist
    
    feature, pred = x
    pred_dist = tf.math.reduce_mean(pred, axis=0)
    feature_dist = tf.math.reduce_mean(feature, axis=0)
    
    #dists = calc_euclidian_dists(z_prototypes, z_query)
    dists = np.array(calc_euclidian_dists(feature_dist, pred_dist))

    def softmax(dist):                         
        a = math.exp(dist[0])

        b = math.exp(dist[1])
            
        c = a/(a+b)
        d = b/(a+b)
        
        prob = tf.concat([c,d], axis=0)
        prob = tf.expand_dims(prob,1)
        
        return prob
    
    prob = softmax(-dists)
    print(prob)
    
    return prob


pred = Lambda(prior_dist)([class_center, query_feature])
combine = Model([sample, query], pred)
optimizer = Adam()
combine.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])


































