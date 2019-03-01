#!/usr/bin/python3
# author: xiangxiang-xu, zhaofeng-shu33
# description: implementation of Alternating Conditional Expectation Algorithm using Neural Network

import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K
import numpy as np


def neg_hscore(x):
    """
    negative hscore calculation
    """
    f = x[0]
    g = x[1]
    f0 = f - K.mean(f, axis = 0)
    g0 = g - K.mean(g, axis = 0)
    corr = tf.reduce_mean(tf.reduce_sum(tf.multiply(f0, g0), 1))
    cov_f = K.dot(K.transpose(f0), f0) / K.cast(K.shape(f0)[0] - 1, dtype = 'float32')
    cov_g = K.dot(K.transpose(g0), g0) / K.cast(K.shape(g0)[0] - 1, dtype = 'float32')
    return - corr + tf.trace(K.dot(cov_f, cov_g)) / 2

def ace_nn(x, y, ns = 1, cat = None, epochs = 300, verbose = False, return_hscore = False):
    ''' 
    Uses the alternating conditional expectations algorithm
    to find the transformations of y and x that maximise the 
    proportion of variation in y explained by x.

    Parameters
    ----------
    x : array_like
        a matrix containing the independent variables.
        each row is an observation of data.
    y : array_like
        a matrix containing the response variables.
        each row is an instance of response.
    epochs : float, optional
        termination threshold (the default is 300). iteration epochs for
        neural network fitting.
    ns : int, optional
        number of eigensolutions (sets of transformations, the default is 1).
    cat : list
        an optional integer vector specifying which variables assume categorical values. 
        nonnegative values in cat refer to columns of the x matrix and -1 refers to the response variable.
    verbose: Integer. 0, 1, or 2. Verbosity mode.
        0 = silent(default), 1 = progress bar, 2 = one line per epoch.   

    Returns
    -------
    tx : array_like
        the transformed x values.
    ty : array_like
        the transformed y values.

    '''

    batch_size = x.shape[0]//8
    hidden_layer_num = 32
    fdim = ns 
    gdim = fdim
    activation_function = 'tanh'
    if(len(x.shape) == 1): # if x is rank-1 matrix
        x_internal = x.reshape([x.shape[0],1])
    else:
        x_internal = x

    if(len(y.shape) == 1): # if y is rank-1 matrix
        y_internal = y.reshape([y.shape[0],1])
    else:
        y_internal = y
    # implementation note: for categorical labels or 
    # input data x, one hot encoding is necessary
    if cat is not None:
        x_cat = []
        y_cat = []
        for i in cat:
            if (i  < -y_internal.shape[1] or i >= x_internal.shape[1]):
                raise ValueError("bad cat= specification")
            if (i < 0):
                y_cat.append(-i-1)
            else: # i > 0, for x response variables
                x_cat.append(i)
        x_encoder = OneHotEncoder(categorical_features=x_cat)
        y_encoder = OneHotEncoder(categorical_features=y_cat)

        x_internal = x_encoder.fit_transform(x_internal)
        y_internal = y_encoder.fit_transform(y_internal)


    input_x = Input(shape = (x_internal.shape[1],))
    f_internal = Dense(hidden_layer_num, activation=activation_function)(input_x)
    f_internal_2 = Dense(hidden_layer_num, activation=activation_function)(f_internal)
    f = Dense(fdim)(f_internal_2)

    input_y = Input(shape = (y_internal.shape[1],))
    g_internal = Dense(hidden_layer_num, activation=activation_function)(input_y)
    g_internal_2 = Dense(hidden_layer_num, activation=activation_function)(g_internal)
    g = Dense(gdim)(g_internal_2)

    loss = Lambda(neg_hscore)([f, g])
    model = Model(inputs = [input_x, input_y], outputs = loss)
    model.compile(optimizer='sgd', loss = lambda y_true,y_pred: y_pred)
    model_f = Model(inputs = input_x, outputs = f)
    model_g = Model(inputs = input_y, outputs = g)
    model.fit([x_internal, y_internal], np.zeros(x_internal.shape), verbose=verbose,
        batch_size = batch_size, epochs = epochs)
    h_score = -model.history.history['loss'][-1] # estimation of 1/2 \norm{\widetilde{B}}_F^2
    t_x = model_f.predict(x_internal)
    t_y = model_g.predict(y_internal)
    if(return_hscore):
        return h_score
    return (t_x, t_y)