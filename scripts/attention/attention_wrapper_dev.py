# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import warnings

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras.engine.topology import _object_list_uid
from keras.utils.generic_utils import has_arg

# Legacy support.
from keras.legacy.layers import Recurrent
from keras.legacy import interfaces

from keras.layers import Input
from keras.layers import (RNN,
                          SimpleRNNCell,
                          LSTMCell, LSTM,
                          GRUCell, GRU,
                          Layer, Input,
                          Wrapper)

from keras.layers.recurrent import *
from keras import Model



class SkeletonCell(Layer):

    def __init__(self, **kwargs):
        super(SkeletonCell, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, states, training=None):
        pass


class ATTRNNCell_Wrapper(Layer):

    def __init__(self, cell, units, attn_size, attn_length,
                 activation='tanh',
                 **kwargs):
        super(ATTRNNCell, self).__init__(**kwargs)
        self._cell = cell
        self._cell.__init__(units, **kwargs)
        self.units = units
        self._attn_size = attn_size
        self._attn_length = attn_length
        self._state_size = self.units
        #self.state_size = (self.units,
        #                   self._attn_size,
        #                   self._attn_size * self._attn_length)
        self.activation = activations.get(activation)

    @property
    def state_size(self):
        #state_size = self.units

        state_size_a = []
        for state_size in (self.units, self.units):
            state_size_a.append(state_size)

        return state_size_a

    #@state_size.setter
    #def state_size(self, state_size):
    #    self.state_size = state_size

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self._cell.build

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

        self.attn_kernel = self.add_weight(shape=(input_dim, self.units),
                                           initializer='uniform',
                                           name='attn_kernel')

    def call(self, inputs, states):
        prev_output = states[0]
        attn_state = states[1:]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        output = self.activation(output)
        return output, [output] + list(attn_state)
