
#########################################
# module: hw02_f19.py
# author: YOUR NAME and YOUR A#.
#########################################

import numpy as np
import pickle
from hw02_f19_data import *

# save() function to save the trained network to a file
def save(ann, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(ann, fp)

# restore() function to restore the file
def load(file_name):
    with open(file_name, 'rb') as fp:
        nn = pickle.load(fp)
    return nn

def sigmoid(x, deriv=False):
    if (deriv == True):
        return np.exp(-x)/((1+np.exp(-x)**2))
    return 1 / (1+np.exp(-x))

def build_nn_wmats(mat_dims):
    np.random.seed(1)
    wmats_result = [None]*(len(mat_dims)-1)
    arrayitter = 0
    for i, j in zip(mat_dims[:-1], mat_dims[1:]):
        temp = np.random.rand(i,j)
        wmats_result[arrayitter] = temp
        arrayitter= arrayitter + 1
    return wmats_result
def build_241_nn():
    return build_nn_wmats((2,4,1))
def build_284_nn():
    return build_nn_wmats((2,8,4))

def build_2683_nn():
    return build_nn_wmats((2,6,8,3))
def build_2542_nn():
    return build_nn_wmats((2,5,4,2))

## Training 3-layer neural net.
## X is the matrix of inputs
## y is the matrix of ground truths.
## build is a nn builder function.
def train_3_layer_nn(numIters, X, y, build):
    W1, W2 = build()
    for i in range(numIters):
        Z2 = np.dot(X, W1)
        layer1 = sigmoid(Z2)
        output = sigmoid(np.dot(layer1, W2))
        output_error = y - output
        output_delta = output_error * sigmoid(output, deriv=True)
        layer1_error = output_delta.dot(W2.T)
        layer1_delta = layer1_error * sigmoid(layer1, deriv=True)
        W2 += layer1.T.dot(output_delta)
        W1 += X.T.dot(layer1_delta)
    return W1, W2


xor_wmats_184 = train_3_layer_nn(100,X1,y_xor,build_284_nn)
xor_wmats_241 = train_3_layer_nn(100,X1,y_xor,build_241_nn)

def train_4_layer_nn(numIters, X, y, build):
    W1, W2, W3 = build()
    for i in range(numIters):
        Z2 = np.dot(X, W1)
        layer1 = sigmoid(Z2)
        layer2 = sigmoid(np.dot(layer1, W2))
        output = sigmoid(np.dot(layer2, W3))
        output_error = y - output
        output_delta = output_error * sigmoid(output, deriv=True)
        layer2_error = output_delta.dot(W3.T)
        layer2_delta = layer2_error * sigmoid(layer2, deriv=True)
        layer1_error = layer2.dot(W2.T)
        layer1_delta = layer1_error * sigmoid(layer1, deriv=True)
        W3 += layer2.T.dot(output_delta)
        W2 += layer1.T.dot(layer2_delta)
        W1 += X.T.dot(layer1_delta)
    return W1, W2, W3


xor_wmats_2542 = train_4_layer_nn(100,X1,y_xor,build_2542_nn)
xor_wmats_91083 = train_4_layer_nn(100,X1,y_xor,build_2683_nn)

def fit_3_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    # your code here
    pass

def fit_4_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    # your code here
    pass

## Remember to state in your comments the structure of each of your
## ANNs (e.g., 2 x 3 x 1 or 2 x 4 x 4 x 1) and how many iterations
## it took you to train it.
        
     




    

    
