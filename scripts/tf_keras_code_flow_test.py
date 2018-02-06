
# coding: utf-8

# In[1]:


import os
import math
import datetime as dt
import itertools as it
from glob import glob
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import tensorflow as tf

import keras
from keras import backend as K
from keras import (regularizers, constraints,
                   initializers, activations)
from keras.layers.recurrent import Recurrent
from keras.models import Sequential, Model
from keras.layers import (SimpleRNN, RNN, LSTM, GRU,
                          Input, Dense, Activation, Lambda,
                          Reshape, Flatten, Permute,
                          Embedding, RepeatVector,
                          TimeDistributed, Bidirectional,
                          dot, multiply, concatenate, merge)
from keras.engine import InputSpec
from keras.callbacks import Callback, LambdaCallback
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils import multi_gpu_model

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.set_printoptions(precision=10)
# # Tensorflow

# # Keras

# In[2]:


def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))


# ---
# ## RNN Code Flow

# ### Forward-Propagation

# In[3]:


units = 2
input_dim = 2

if input_dim == 1:
    data_X = np.array([[.3],
                       [.7],
                       [.4]
                      ]).reshape(1, 3, 1).astype('float32')
else:
    data_X = np.array([[.3, .8],
                       [.7, .6],
                       [.4, 0.]
                      ]).reshape(1, 3, 2).astype('float32')

print(data_X.shape)
pprint(data_X)


# #### `return_sequences=False`

# In[4]:


units = units
#init_state = np.zeros((1, units), dtype=np.float32)
#init_state = [K.zeros(init_state.shape, dtype=np.float32, name=None)]
n_sample, seq_len, input_dim = data_X.shape

# Define an input sequence and process it.
_input_layer = Input(shape=(seq_len, input_dim), name='Input_layer')

_rnn_layer = SimpleRNN(units, return_sequences=False,
                       return_state=True,
                       name='RNN_layer')
outputs, states = _rnn_layer(_input_layer)

#_output_layer = Dense(1, activation='softmax', name='Output_layer')
#_output_layer = Activation('softmax', name='Output_layer')
#_outputs = _output_layer(outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model(inputs=_input_layer, outputs=[outputs, states])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

pprint(model.weights)
model.summary()


# * Create Session Error Exception

# In[5]:


try:
    model.get_weights()
except:
    pass


# In[6]:


kernel, recurrent_kernel, bias = _rnn_layer.get_weights()
print(kernel, recurrent_kernel, bias, sep='\n')
print('Kernel_shape:', kernel.shape)
print('Recurrent_Kernel_shape:', recurrent_kernel.shape)
print('Bias_shape:', bias.shape)


# In[7]:


_rnn_layer.get_initial_state(data_X)


# In[8]:


weight = ['h']

my_kernel = (np.arange(input_dim * units* len(weight)) * .1 + .2).reshape(input_dim, units* len(weight))
my_recurrent_kernel = (np.arange(units * units * len(weight)) * .01 - .05).reshape(units, units * len(weight))
my_bias = np.zeros(units * len(weight)).reshape(units * len(weight), )

print(my_kernel, my_recurrent_kernel, my_bias, sep='\n')

_rnn_layer.set_weights([my_kernel, my_recurrent_kernel, my_bias])


# In[9]:


kernel, recurrent_kernel, bias = model.get_weights()
print(kernel, recurrent_kernel, bias, sep='\n')

return_sequence = False
return_state = True


sequence = []
for _ in range(len(data_X[0])):
    
    inputs = data_X[0][_]
    inputs = inputs.astype(np.float32)
    print('inputs')
    print(inputs.shape)
    pprint(inputs)

    print('h -----------')
    h = np.dot(inputs, kernel) + bias
    h = h.astype(np.float32)
    print(h.shape)
    pprint(h)
    #print('\nh:', h)

    if _ == 0:
        initial_state = [np.zeros(units, dtype=np.float32)]
        states = initial_state
    else:
        states = [output]
    prev_output = states[0]
    print('prev --------')
    print(prev_output.shape)
    pprint(prev_output)
    #print('\nprev:', prev_output)

    print('a -----------')
    a = np.dot(prev_output, recurrent_kernel)
    a = a.astype(np.float32)
    print(a.shape)
    pprint(a)

    print('output ------')
    output = h + a
    output = output.astype(np.float32)
    print(output.shape)
    pprint(output)
    
    
    print('\nstep (%s)' % _ + '='*37)
    print('h\t\t:', h)
    print('prev\t\t:', prev_output)
    print('output(%s)\t: %s' % (_, output))
    print('='*45 + '\n')
    
    sequence.append(output)
    
    if return_sequence:
        result = sequence
    else:
        result = sequence[-1]
    
    if return_state:
        state = output
    else:
        state = []
        
        
result = np.expand_dims(np.stack(result), axis=0)
state = np.expand_dims(np.array(state), axis=0)

print('\nResult: [Output, State]')
print('\n=== Numpy ===')
print([result, state])
print('\n=== Keras ===')
print(model.predict(data_X))


# #### `return_sequences=True`

# In[10]:


units = units
n_sample, seq_len, input_dim = data_X.shape

# Define an input sequence and process it.
_input_layer = Input(shape=(seq_len, input_dim), name='Input_layer')

_rnn_layer = SimpleRNN(units, return_sequences=True, return_state=True, name='RNN_layer')
outputs, states = _rnn_layer(_input_layer)

#_output_layer = Dense(1, activation='softmax', name='Output_layer')
#_output_layer = Activation('softmax', name='Output_layer')
#_outputs = _output_layer(outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model(inputs=_input_layer, outputs=[outputs, states])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

pprint(model.weights)
model.summary()


# In[11]:


weight = ['h']

my_kernel = (np.arange(input_dim * units* len(weight)) * .1 + .2).reshape(input_dim, units* len(weight)).astype(np.float32)
my_recurrent_kernel = (np.arange(units * units * len(weight)) * .01 - .05).reshape(units, units * len(weight)).astype(np.float32)
my_bias = np.zeros(units * len(weight)).reshape(units * len(weight), ).astype(np.float32)

print(my_kernel, my_recurrent_kernel, my_bias, sep='\n')

_rnn_layer.set_weights([my_kernel, my_recurrent_kernel, my_bias])


# In[12]:


kernel, recurrent_kernel, bias = model.get_weights()
print(kernel, recurrent_kernel, bias, sep='\n')

return_sequence = True
return_state = True


sequence = []
for _ in range(len(data_X[0])):
    
    inputs = data_X[0][_]
    inputs = inputs.astype(np.float32)
    print('inputs')
    print(inputs.shape)
    pprint(inputs)

    print('h -----------')
    h = np.dot(inputs, kernel) + bias
    h = h.astype(np.float32)
    print(h.shape)
    pprint(h)
    #print('\nh:', h)

    if _ == 0:
        initial_state = [np.zeros(units, dtype=np.float32)]
        states = initial_state
    else:
        states = [output]
    prev_output = states[0]
    print('prev --------')
    print(prev_output.shape)
    pprint(prev_output)
    #print('\nprev:', prev_output)

    print('a -----------')
    a = np.dot(prev_output, recurrent_kernel)
    a = a.astype(np.float32)
    print(a.shape)
    pprint(a)

    print('output ------')
    output = h + a
    output = output.astype(np.float32)
    print(output.shape)
    pprint(output)
    
    
    print('\nstep (%s)' % _ + '='*37)
    print('h\t\t:', h)
    print('prev\t\t:', prev_output)
    print('output(%s)\t: %s' % (_, output))
    print('='*45 + '\n')
    
    sequence.append(output)
    
    if return_sequence:
        result = sequence
    else:
        result = sequence[-1]
    
    if return_state:
        state = output
    else:
        state = []
        
        
result = np.expand_dims(np.stack(result), axis=0)
state = np.expand_dims(np.array(state), axis=0)

print('\nResult: [Output, State]')
print('\n=== Numpy ===')
print([result, state])
print('\n=== Keras ===')
print(model.predict(data_X))


# ### Back-Propagation

# In[39]:


from scipy.integrate import odeint
from scipy.misc import derivative


# In[34]:


x = .2
f = lambda x: -.3*x + .1
g = lambda x: .4*x - .4

y = g(x)
z = f(g(x))

print('X: %s, Y: %s, Z: %s' % (x, y, z))


# In[40]:


derivative(f, x)


# In[44]:


x = 5
f = lambda x: 3*(x**2) - 1

y = f(x)

print('X: %s, Y: %s' % (x, y))


# In[45]:


derivative(f, x)


# In[50]:


def sigmo(x):
    return 1 / (1 + np.exp(-x))


# In[38]:


odeint(f, y, .2)


# ---
# ## GRU Code Flow

# In[13]:


units = 2
input_dim = 2

if input_dim == 1:
    data_X = np.array([[.2],
                       [.5],
                       [.4]
                      ]).reshape(1, 3, 1).astype('float32')
else:
    data_X = np.array([[.2, .1],
                       [.5, .3],
                       [.4, 0.]
                      ]).reshape(1, 3, 2).astype('float32')

print(data_X.shape)
pprint(data_X)


# #### `return_sequences=False`

# In[14]:


units = units
n_sample, seq_len, input_dim = data_X.shape

# Define an input sequence and process it.
_input_layer = Input(shape=(seq_len, input_dim), name='Input_layer')

_rnn_layer = GRU(units, return_sequences=False, return_state=True, name='RNN_layer')
outputs, states = _rnn_layer(_input_layer)

#_output_layer = Dense(1, activation='softmax', name='Output_layer')
#_output_layer = Activation('softmax', name='Output_layer')
#_outputs = _output_layer(outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model(inputs=_input_layer, outputs=[outputs, states])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

pprint(model.weights)
model.summary()


# In[15]:


kernel, recurrent_kernel, bias = _rnn_layer.get_weights()
print(kernel, recurrent_kernel, bias, sep='\n')
print('Kernel_shape:', kernel.shape)
print('Recurrent_Kernel_shape:', recurrent_kernel.shape)
print('Bias_shape:', bias.shape)


# In[16]:


weight = ['z', 'r', 'h']

my_kernel = (np.arange(input_dim * units* len(weight)) * .1 + .2).reshape(input_dim, units* len(weight)).astype(np.float32)
my_recurrent_kernel = (np.arange(units * units * len(weight)) * .01 - .05).reshape(units, units * len(weight)).astype(np.float32)
my_bias = np.zeros(units * len(weight)).reshape(units * len(weight), ).astype(np.float32)

print(my_kernel, my_recurrent_kernel, my_bias, sep='\n')

_rnn_layer.set_weights([my_kernel, my_recurrent_kernel, my_bias])


# In[17]:


kernel, recurrent_kernel, bias = model.get_weights()
print(kernel, recurrent_kernel, bias, sep='\n')

return_sequence = False
return_state = True

"""
activation = np.tanh
recurrent_activation = sigmoid
"""

kernel_z = kernel[:,          : units * 1]
kernel_r = kernel[:, units * 1: units * 2]
kernel_h = kernel[:, units * 2:]

recurrent_kernel_z = recurrent_kernel[:,          : units * 1]
recurrent_kernel_r = recurrent_kernel[:, units * 1: units * 2]
recurrent_kernel_h = recurrent_kernel[:, units * 2:]

bias_z = bias[         : units * 1]
bias_r = bias[units    : units * 2]
bias_h = bias[units * 2:]

sequence = []
for _ in range(len(data_X[0])):
    
    inputs = data_X[0][_]
    inputs = inputs.astype(np.float32)
    print('inputs')
    print(inputs.shape)
    pprint(inputs)

    print('x -----------')
    print('dropout(input) time')
    x_z = np.dot(inputs, kernel_z) + bias_z
    x_r = np.dot(inputs, kernel_r) + bias_r
    x_h = np.dot(inputs, kernel_h) + bias_h

    if _ == 0:
        initial_state = [np.zeros(units, dtype=np.float32)]
        states = initial_state
    else:
        states = [output]
    h_tm1 = prev_output = states[0]
    print('prev --------')
    print(prev_output.shape)
    pprint(prev_output)
    #print('\nprev:', prev_output)
    
    print('gate --------')
    print('recurrent dropout(h_tm1) time')
    z = sigmoid(x_z + np.dot(h_tm1, recurrent_kernel_z))
    r = sigmoid(x_r + np.dot(h_tm1, recurrent_kernel_r))
    hh = np.tanh(x_h + np.dot(h_tm1, recurrent_kernel_h))
    
    h = z * h_tm1 + (1 - z) * hh
    h = h.astype(np.float32)
    print(h.shape)
    pprint(h)
    #print('\nh:', h)

    print('a -----------')
    a = np.dot(prev_output, recurrent_kernel)
    a = a.astype(np.float32)
    print(a.shape)
    pprint(a)

    print('output ------')
    output = h
    output = output.astype(np.float32)
    print(output.shape)
    pprint(output)
    
    
    print('\nstep (%s)' % _ + '='*37)
    print('h\t\t:', h)
    print('prev\t\t:', prev_output)
    print('output(%s)\t: %s' % (_, output))
    print('='*45 + '\n')
    
    sequence.append(output)
    
    if return_sequence:
        result = sequence
    else:
        result = sequence[-1]
    
    if return_state:
        state = output
    else:
        state = []
        
        
result = np.expand_dims(np.stack(result), axis=0)
state = np.expand_dims(np.array(state), axis=0)

print('\nResult: [Output, State]')
print('\n=== Numpy ===')
print([result, state])
print('\n=== Keras ===')
print(model.predict(data_X))


# #### `return_sequences=True`

# In[18]:


units = units
n_sample, seq_len, input_dim = data_X.shape

# Define an input sequence and process it.
_input_layer = Input(shape=(seq_len, input_dim), name='Input_layer')

_rnn_layer = GRU(units, return_sequences=True, return_state=True, name='RNN_layer')
outputs, states = _rnn_layer(_input_layer)

#_output_layer = Dense(1, activation='softmax', name='Output_layer')
#_output_layer = Activation('softmax', name='Output_layer')
#_outputs = _output_layer(outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model(inputs=_input_layer, outputs=[outputs, states])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

pprint(model.weights)
model.summary()


# In[19]:


weight = ['z', 'r', 'h']

my_kernel = (np.arange(input_dim * units* len(weight)) * .1 + .2).reshape(input_dim, units* len(weight)).astype(np.float32)
my_recurrent_kernel = (np.arange(units * units * len(weight)) * .01 - .05).reshape(units, units * len(weight)).astype(np.float32)
my_bias = np.zeros(units * len(weight)).reshape(units * len(weight), ).astype(np.float32)

print(my_kernel, my_recurrent_kernel, my_bias, sep='\n')

_rnn_layer.set_weights([my_kernel, my_recurrent_kernel, my_bias])


# In[20]:


kernel, recurrent_kernel, bias = model.get_weights()
print(kernel, recurrent_kernel, bias, sep='\n')

return_sequence = True
return_state = True

"""
activation = np.tanh
recurrent_activation = sigmoid
"""

kernel_z = kernel[:,          : units * 1]
kernel_r = kernel[:, units * 1: units * 2]
kernel_h = kernel[:, units * 2:]

recurrent_kernel_z = recurrent_kernel[:,          : units * 1]
recurrent_kernel_r = recurrent_kernel[:, units * 1: units * 2]
recurrent_kernel_h = recurrent_kernel[:, units * 2:]

bias_z = bias[         : units * 1]
bias_r = bias[units    : units * 2]
bias_h = bias[units * 2:]

sequence = []
for _ in range(len(data_X[0])):
    
    inputs = data_X[0][_]
    inputs = inputs.astype(np.float32)
    print('inputs')
    print(inputs.shape)
    pprint(inputs)

    print('x -----------')
    print('dropout(input) time')
    x_z = np.dot(inputs, kernel_z) + bias_z
    x_r = np.dot(inputs, kernel_r) + bias_r
    x_h = np.dot(inputs, kernel_h) + bias_h

    if _ == 0:
        initial_state = [np.zeros(units, dtype=np.float32)]
        states = initial_state
    else:
        states = [output]
    h_tm1 = prev_output = states[0]
    print('prev --------')
    print(prev_output.shape)
    pprint(prev_output)
    #print('\nprev:', prev_output)
    
    print('gate --------')
    print('recurrent dropout(h_tm1) time')
    z = sigmoid(x_z + np.dot(h_tm1, recurrent_kernel_z))
    r = sigmoid(x_r + np.dot(h_tm1, recurrent_kernel_r))
    hh = np.tanh(x_h + np.dot(h_tm1, recurrent_kernel_h))
    
    h = z * h_tm1 + (1 - z) * hh
    h = h.astype(np.float32)
    print(h.shape)
    pprint(h)
    #print('\nh:', h)

    print('a -----------')
    a = np.dot(prev_output, recurrent_kernel)
    a = a.astype(np.float32)
    print(a.shape)
    pprint(a)

    print('output ------')
    output = h
    output = output.astype(np.float32)
    print(output.shape)
    pprint(output)
    
    
    print('\nstep (%s)' % _ + '='*37)
    print('h\t\t:', h)
    print('prev\t\t:', prev_output)
    print('output(%s)\t: %s' % (_, output))
    print('='*45 + '\n')
    
    sequence.append(output)
    
    if return_sequence:
        result = sequence
    else:
        result = sequence[-1]
    
    if return_state:
        state = output
    else:
        state = []
        
        
result = np.expand_dims(np.stack(result), axis=0)
state = np.expand_dims(np.array(state), axis=0)

print('\nResult: [Output, State]')
print('\n=== Numpy ===')
print([result, state])
print('\n=== Keras ===')
print(model.predict(data_X))


# ---
# ## LSTM Code Flow

# In[21]:


units = 2
input_dim = 1

if input_dim == 1:
    data_X = np.array([[.2],
                       [.5],
                       [.4]
                      ]).reshape(1, 3, 1).astype('float32')
else:
    data_X = np.array([[.2, .1],
                       [.5, .3],
                       [.4, 0.]
                      ]).reshape(1, 3, 2).astype('float32')

print(data_X.shape)
pprint(data_X)


# #### `return_sequences=False`

# In[22]:


units = units
n_sample, seq_len, input_dim = data_X.shape

# Define an input sequence and process it.
_input_layer = Input(shape=(seq_len, input_dim), name='Input_layer')

_rnn_layer = LSTM(units, return_sequences=False, return_state=True, name='RNN_layer')
outputs, state_h, state_c = _rnn_layer(_input_layer)

#_output_layer = Dense(1, activation='softmax', name='Output_layer')
#_output_layer = Activation('softmax', name='Output_layer')
#_outputs = _output_layer(outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model(inputs=_input_layer, outputs=[outputs, state_h, state_c])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

pprint(model.weights)
model.summary()


# In[23]:


kernel, recurrent_kernel, bias = _rnn_layer.get_weights()
print(kernel, recurrent_kernel, bias, sep='\n')
print('Kernel_shape:', kernel.shape)
print('Recurrent_Kernel_shape:', recurrent_kernel.shape)
print('Bias_shape:', bias.shape)


# In[24]:


weight = ['i', 'f', 'c', 'o']

my_kernel = (np.arange(input_dim * units* len(weight)) * .1 + .2).reshape(input_dim, units* len(weight)).astype(np.float32)
my_recurrent_kernel = (np.arange(units * units * len(weight)) * .01 - .05).reshape(units, units * len(weight)).astype(np.float32)
my_bias = np.zeros(units * len(weight)).reshape(units * len(weight), ).astype(np.float32)

print(my_kernel, my_recurrent_kernel, my_bias, sep='\n')

_rnn_layer.set_weights([my_kernel, my_recurrent_kernel, my_bias])


# In[25]:


kernel, recurrent_kernel, bias = model.get_weights()
print(kernel, recurrent_kernel, bias, sep='\n')

return_sequence = False
return_state = True

"""
activation = np.tanh
recurrent_activation = sigmoid
"""

kernel_i = kernel[:,           :units * 1]
kernel_f = kernel[:, units * 1: units * 2]
kernel_c = kernel[:, units * 2: units * 3]
kernel_o = kernel[:, units * 3:]

recurrent_kernel_i = recurrent_kernel[:,          : units * 1]
recurrent_kernel_f = recurrent_kernel[:, units * 1: units * 2]
recurrent_kernel_c = recurrent_kernel[:, units * 2: units * 3]
recurrent_kernel_o = recurrent_kernel[:, units * 3:]

bias_i = bias[         : units * 1]
bias_f = bias[units * 1: units * 2]
bias_c = bias[units * 2: units * 3]
bias_o = bias[units * 3:          ]

sequence = []
for _ in range(len(data_X[0])):
    
    inputs = data_X[0][_]
    inputs = inputs.astype(np.float32)
    print('inputs')
    print(inputs.shape)
    pprint(inputs)

    print('x -----------')
    print('dropout(input) time')
    x_i = np.dot(inputs, kernel_i) + bias_i
    x_f = np.dot(inputs, kernel_f) + bias_f
    x_c = np.dot(inputs, kernel_c) + bias_c
    x_o = np.dot(inputs, kernel_o) + bias_o

    if _ == 0:
        initial_state = [np.zeros(units, dtype=np.float32)] * 2
        states = initial_state
    else:
        states = [h, c]
    h_tm1 = prev_output = states[0]
    c_tm1 =               states[1]
    print('prev --------')
    print(prev_output.shape)
    pprint(prev_output)
    #print('\nprev:', prev_output)
    
    print('gate --------')
    print('recurrent dropout(h_tm1) time')
    i = sigmoid(x_i + np.dot(h_tm1, recurrent_kernel_i))
    f = sigmoid(x_f + np.dot(h_tm1, recurrent_kernel_f))
    c = f * c_tm1 +        i * sigmoid(x_c + np.dot(h_tm1, recurrent_kernel_c))
    o = np.tanh(x_o + np.dot(h_tm1, recurrent_kernel_o))
    
    h = o * np.tanh(c)
    h = h.astype(np.float32)
    print(h.shape)
    pprint(h)
    #print('\nh:', h)

    print('output ------')
    output = h
    output = output.astype(np.float32)
    print(output.shape)
    pprint(output)
    
    
    print('\nstep (%s)' % _ + '='*37)
    print('h\t\t:', h)
    print('prev\t\t:', prev_output)
    print('output(%s)\t: %s' % (_, output))
    print('='*45 + '\n')
    
    sequence.append(output)
    
    if return_sequence:
        result = sequence
    else:
        result = sequence[-1]
    
    if return_state:
        state = [h, c]
    else:
        state = []
        
        
result = np.expand_dims(np.stack(result), axis=0)
state = np.expand_dims(np.array(state), axis=0)

print('\nResult: [Output, State]')
print('\n=== Numpy ===')
print([result, state])
print('\n=== Keras ===')
print(model.predict(data_X))


# #### `return_sequences=True`

# In[26]:


units = units
n_sample, seq_len, input_dim = data_X.shape

# Define an input sequence and process it.
_input_layer = Input(shape=(seq_len, input_dim), name='Input_layer')

_rnn_layer = LSTM(units, return_sequences=True, return_state=True, name='RNN_layer')
outputs, state_h, state_c = _rnn_layer(_input_layer)

#_output_layer = Dense(1, activation='softmax', name='Output_layer')
#_output_layer = Activation('softmax', name='Output_layer')
#_outputs = _output_layer(outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model(inputs=_input_layer, outputs=[outputs, state_h, state_c])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

pprint(model.weights)
model.summary()


# In[27]:


weight = ['i', 'f', 'c', 'o']

my_kernel = (np.arange(input_dim * units* len(weight)) * .1 + .2).reshape(input_dim, units* len(weight)).astype(np.float32)
my_recurrent_kernel = (np.arange(units * units * len(weight)) * .01 - .05).reshape(units, units * len(weight)).astype(np.float32)
my_bias = np.zeros(units * len(weight)).reshape(units * len(weight), ).astype(np.float32)

print(my_kernel, my_recurrent_kernel, my_bias, sep='\n')

_rnn_layer.set_weights([my_kernel, my_recurrent_kernel, my_bias])


# In[28]:


kernel, recurrent_kernel, bias = model.get_weights()
print(kernel, recurrent_kernel, bias, sep='\n')

return_sequence = True
return_state = True

"""
activation = np.tanh
recurrent_activation = sigmoid
"""

kernel_i = kernel[:,           :units * 1]
kernel_f = kernel[:, units * 1: units * 2]
kernel_c = kernel[:, units * 2: units * 3]
kernel_o = kernel[:, units * 3:]

recurrent_kernel_i = recurrent_kernel[:,          : units * 1]
recurrent_kernel_f = recurrent_kernel[:, units * 1: units * 2]
recurrent_kernel_c = recurrent_kernel[:, units * 2: units * 3]
recurrent_kernel_o = recurrent_kernel[:, units * 3:]

bias_i = bias[         : units * 1]
bias_f = bias[units * 1: units * 2]
bias_c = bias[units * 2: units * 3]
bias_o = bias[units * 3:          ]

sequence = []
for _ in range(len(data_X[0])):
    
    inputs = data_X[0][_]
    inputs = inputs.astype(np.float32)
    print('inputs')
    print(inputs.shape)
    pprint(inputs)

    print('x -----------')
    print('dropout(input) time')
    x_i = np.dot(inputs, kernel_i) + bias_i
    x_f = np.dot(inputs, kernel_f) + bias_f
    x_c = np.dot(inputs, kernel_c) + bias_c
    x_o = np.dot(inputs, kernel_o) + bias_o

    if _ == 0:
        initial_state = [np.zeros(units, dtype=np.float32)] * 2
        states = initial_state
    else:
        states = [h, c]
    h_tm1 = prev_output = states[0]
    c_tm1 =               states[1]
    print('prev --------')
    print(prev_output.shape)
    pprint(prev_output)
    #print('\nprev:', prev_output)
    
    print('gate --------')
    print('recurrent dropout(h_tm1) time')
    i = sigmoid(x_i + np.dot(h_tm1, recurrent_kernel_i))
    f = sigmoid(x_f + np.dot(h_tm1, recurrent_kernel_f))
    c = f * c_tm1 +        i * sigmoid(x_c + np.dot(h_tm1, recurrent_kernel_c))
    o = np.tanh(x_o + np.dot(h_tm1, recurrent_kernel_o))
    
    h = o * np.tanh(c)
    h = h.astype(np.float32)
    print(h.shape)
    pprint(h)
    #print('\nh:', h)

    print('output ------')
    output = h
    output = output.astype(np.float32)
    print(output.shape)
    pprint(output)
    
    
    print('\nstep (%s)' % _ + '='*37)
    print('h\t\t:', h)
    print('prev\t\t:', prev_output)
    print('output(%s)\t: %s' % (_, output))
    print('='*45 + '\n')
    
    sequence.append(output)
    
    if return_sequence:
        result = sequence
    else:
        result = sequence[-1]
    
    if return_state:
        state = [h, c]
    else:
        state = []
        
        
result = np.expand_dims(np.stack(result), axis=0)
state = np.expand_dims(np.array(state), axis=0)

print('\nResult: [Output, State]')
print('\n=== Numpy ===')
print([result, state])
print('\n=== Keras ===')
print(model.predict(data_X))


# Done.
