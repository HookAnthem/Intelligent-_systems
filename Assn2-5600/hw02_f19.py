
#########################################
# module: hw02_f19.py
# author: Zachary Hook A02231717
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

def build_461_nn():
    return build_nn_wmats((4,6,1))
def build_2151_nn():
    return build_nn_wmats((2,15,1))
def build_281_nn():
    return build_nn_wmats((2,8,1))
def build_181_nn():
    return build_nn_wmats((1,8,1))
def build_2681_nn():
    return build_nn_wmats((2,6,8,1))
def build_2504001_nn():
    return build_nn_wmats((2,50,400,1))
def build_2641_nn():
    return build_nn_wmats((2,5,9,1))
def build_1641_nn():
    return build_nn_wmats((1,6,4,1))
def build_45081_nn():
    return build_nn_wmats((4,50,8,1))
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
        layer1_error = layer2_delta.dot(W2.T)
        layer1_delta = layer1_error * sigmoid(layer1, deriv=True)
        W3 += layer2.T.dot(output_delta)
        W2 += layer1.T.dot(layer2_delta)
        W1 += X.T.dot(layer1_delta)
    return W1, W2, W3

def fit_3_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    W1, W2 = wmats
    Z2 = np.dot(x, W1)
    layer1 = sigmoid(Z2)
    output = sigmoid(np.dot(layer1, W2))
    if thresh_flag == True:
        temp =[0]*len(x)
        array_itter = 0
        for i in x:
            if output[array_itter] > thresh:
                temp[array_itter] = 1
            array_itter += 1
        print(temp)
        return temp
    print(output)
    return output

def fit_4_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    W1, W2, W3 = wmats
    Z2 = np.dot(x, W1)
    layer1 = sigmoid(Z2)
    layer2 = sigmoid(np.dot(layer1, W2))
    output = sigmoid(np.dot(layer2, W3))
    if thresh_flag == True:
        temp = [0] * len(x)
        array_itter = 0
        for i in x:
            if output[array_itter] > thresh:
                temp[array_itter] = 1
            array_itter += 1
        print(temp)
        return temp
    print(output)
    return output
    # your code here
    pass

## Remember to state in your comments the structure of each of your
## ANNs (e.g., 2 x 3 x 1 or 2 x 4 x 4 x 1) and how many iterations
## it took you to train it.

#Building a 2x8x1 it took 700 iterations to train
xor_wmats_281 = train_3_layer_nn(700, X1, y_xor, build_281_nn)
save(xor_wmats_281, 'D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\xor_3_layer_ann.pck')
loaded_wmats_xor_3 = load('D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\xor_3_layer_ann.pck')

#building a 2x8x1 it took 700 iterations to train
or_wmats_281 = train_3_layer_nn(700, X1, y_or, build_281_nn)
save(or_wmats_281, 'D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\or_3_layer_ann.pck')
loaded_wmats_or_3 = load('D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\or_3_layer_ann.pck')

#Building a 2x15x1 it took 50 iterations to train
AND_wmats_2151 = train_3_layer_nn(50, X1, y_and, build_2151_nn)
save(AND_wmats_2151, 'D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\and_3_layer_ann.pck')
loaded_wmats_and_3 = load('D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\and_3_layer_ann.pck')

#
not_wmats_181 = train_3_layer_nn(1000, X2, y_not, build_181_nn)
save(not_wmats_181, 'D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\not_3_layer_ann.pck')
loaded_wmats_not_3 = load('D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\not_3_layer_ann.pck')

#Building a 4x6x1 it took 1000 iterations to train
bool_wmats_461 = train_3_layer_nn(1000, X4, bool_exp, build_461_nn)
save(bool_wmats_461, 'D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\bool_3_layer_ann.pck')
loaded_wmats_bool_3 = load('D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\bool_3_layer_ann.pck')

#Building a 2x6x8x1 it took 2000 iterations to train
xor_wmats_2681 = train_4_layer_nn(2000, X1, y_xor, build_2681_nn)
save(xor_wmats_2681, "D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\xor_4_layer_ann.pck")
loaded_wmats_xor_4 = load("D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\xor_4_layer_ann.pck")

#building a 2x50x400x1 it took 5000 iterations to train
or_wmats_2504001 = train_4_layer_nn(5000, X1, y_or, build_2504001_nn)
save(or_wmats_2504001, 'D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\or_4_layer_ann.pck')
loaded_wmats_or_4 = load('D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\or_4_layer_ann.pck')

#Building a 2x6x4x1 it took 50 iterations to train
AND_wmats_2641 = train_4_layer_nn(50, X1, y_and, build_2641_nn)
save(AND_wmats_2641, 'D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\and_4_layer_ann.pck')
loaded_wmats_and_4 = load('D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\and_4_layer_ann.pck')

#Building  a 1X6X4X1 it took 1000 iterations to train
not_wmats_1641 = train_4_layer_nn(1000, X2, y_not, build_1641_nn)
save(not_wmats_1641, 'D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\not_4_layer_ann.pck')
loaded_wmats_not_4 = load('D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\not_4_layer_ann.pck')

#Building a 4X50X8X1 it takes 1000000 iterations to train it
bool_wmats_45081 = train_4_layer_nn(1000000, X4, bool_exp, build_45081_nn)
save(bool_wmats_45081, 'D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\bool_4_layer_ann.pck')
loaded_wmats_bool_4 = load('D:\Zach\CS5600\Intelligent-_systems\Assn2-5600\\bool_4_layer_ann.pck')

print("3 layer xor")
fitted_xor_281 = fit_3_layer_nn(X1, loaded_wmats_xor_3, .5, thresh_flag=True)

print("3 layer or")
fitted_or_281 = fit_3_layer_nn(X1, loaded_wmats_or_3, .5, thresh_flag=True)

print("3 layer and")
fitted_AND_2401 = fit_3_layer_nn(X1, loaded_wmats_and_3, .5, thresh_flag=True)

print("3 layer not")
fitted_not_181 = fit_3_layer_nn(X2, loaded_wmats_not_3, .5, thresh_flag=True)

print("3 layer boolean")
fitted_bool_461 = fit_3_layer_nn(X4, loaded_wmats_bool_3, .5, thresh_flag=True)

print("4 layer or")
fitted_or_2504001 = fit_4_layer_nn(X1, loaded_wmats_or_4, .5, thresh_flag=True)

print("4 layer xor")
fitted_xor_2681 = fit_4_layer_nn(X1, loaded_wmats_xor_4, .5, thresh_flag=True)

print("4 layer and")
fitted_AND_2641 = fit_4_layer_nn(X1, loaded_wmats_and_4, .27, thresh_flag=True)

print("4 layer not")
fitted_not_1641 = fit_4_layer_nn(X2, loaded_wmats_not_4, .27, thresh_flag=True)

print("4 layer bool")
fitted_not_4721 = fit_4_layer_nn(X4, loaded_wmats_bool_4, .6, thresh_flag=True)


