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
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
warnings.filterwarnings("ignore")

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
    dist = tf.math.multiply(dist,10**5)
    dist = tf.expand_dims(dist,0)
    #print(dist)
    return dist

class Prototypical(Model):
    """
    Implemenation of Prototypical Network.
    """
    def __init__(self, w, h, c):
        """
        Args:
            n_support (int): number of support examples.
            n_query (int): number of query examples.
            w (int): image width .
            h (int): image height.
            c (int): number of channels.
        """
        super(Prototypical, self).__init__()
        self.w, self.h, self.c = w, h, c

        # Encoder as ResNet like CNN with 4 blocks
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)), Flatten()]
        )  
           
        
    def call(self):
        
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
        
        z = self.encoder(x_train1[0:400])
        #print(z)
        z1 = self.encoder(x_train1[600:1000])
    
        z_prototypes = tf.math.reduce_mean(z, axis=0)
        #print(z_prototypes)
        z_prototypes1 = tf.math.reduce_mean(z1, axis=0)
        
        z_query = self.encoder(x_test1[30:31])
        z_query1 = self.encoder(x_test1[119:120])
        #print(z_query)
        #z_query = tf.math.reduce_mean(z_query, axis=1)
        #print(z_query[1])
        
        #dists = calc_euclidian_dists(z_prototypes, z_query)
        dists = np.array([calc_euclidian_dists(z_prototypes, z_query[i])  
                          for i in range(len(z_query))])
        dists1 = np.array([calc_euclidian_dists(z_prototypes1, z_query[i])  
                          for i in range(len(z_query))])
        
        # print(dists)
        
        # dists1 = np.array([calc_euclidian_dists(z_prototypes1, z_query1[i])  
        #                   for i in range(len(z_query))])
        
        # print("""\n break \n""")
        # print(dists1)
        
        dist = tf.concat([dists,dists1], axis=0)
        print(dist)
        
        def softmax(dist):                         
            a = math.exp(dist[0])
    
            b = math.exp(dist[1])
                
            c = a/(a+b)
            d = b/(a+b)
            
            prob = tf.concat([c,d], axis=0)
            prob = tf.expand_dims(prob,1)
            
            return prob
        
        prob = softmax(-dist)
        print(prob)
                
        
        #log_p_y = [tf.nn.log_softmax(-dist[i], axis=-1) for i in range(len(dist))]
        # log_p_y = tf.nn.log_softmax(dist, axis=0)
        # log_p_y = np.array(log_p_y)
        #print(log_p_y)
        
        #loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))
        
        labels = tf.one_hot([0, 1], 1)
        print(labels)

        cce = tf.keras.losses.CategoricalCrossentropy()

        loss = cce(labels, prob).numpy()     
        
       # train = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        
        #Model([left_input, right_input], prediction)
        # train = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        
        # with tf.Session() as sess:
        #   sess.run( tf.global_variables_initializer() )
        #   for step in range(50) :
        #     sess.run(train)
        
        # training step : gradient decent (1.0) to minimize loss
        tf.optimizers.SGD (learning_rate=0.001)
        

        return print(loss)
                
             
net = Prototypical(32,32,3).call()
        

        
        
        
        
        
        
        
        
        