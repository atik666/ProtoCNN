import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import os
import cv2
from sklearn.model_selection import train_test_split

def train_test_split_tensors(X, y, **options):
    """
    encapsulation for the sklearn.model_selection.train_test_split function
    in order to split tensors objects and return tensors as output

    :param X: tensorflow.Tensor object
    :param y: tensorflow.Tensor object
    :dict **options: typical sklearn options are available, such as test_size and train_size
    """

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
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
    return tf.reduce_mean(tf.math.pow(x - y, 2), 2)

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
                  
    path = '/home/atik/Documents/UMAML_FSL/data/train/n01532829'
    
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
    
    X1 = import_img('/home/atik/Documents/UMAML_FSL/data/train/n01532829')
    y1 = tf.convert_to_tensor(np.zeros(len(X1)).astype('float32'))
    X2 = import_img('/home/atik/Documents/UMAML_FSL/data/train/n02113712')
    y2 = tf.convert_to_tensor(np.zeros(len(X2)).astype('float32')+1)
        
    X = tf.concat([X1, X2], axis=0)
    y = tf.concat([y1, y2], axis=0)
    
    x_train1, x_test1, y_train1, y_test1 = train_test_split_tensors(X1, y1, test_size=0.1, shuffle=False)
    
    z = self.encoder(training_data)
    
    
    z_prototypes = tf.math.reduce_mean(z, axis=1)
        
    return z_prototypes
                
             
net = Prototypical(32,32,3)
        
        
        
        
        
        
        
        
        
        
        
        
        