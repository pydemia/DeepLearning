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
from ..engine.topology import _object_list_uid
from keras.utils.generic_utils import has_arg

# Legacy support.
from keras.legacy.layers import Recurrent
from keras.legacy import interfaces

from keras.layers import (RNN, GRUCell, GRU,
                          _generate_dropout_ones,
                          _generate_dropout_mask)


def _generate_dropout_ones(inputs, dims):
    # Currently, CTNK can't instantiate `ones` with symbolic shapes.
    # Will update workaround once CTNK supports it.
    if K.backend() == 'cntk':
        ones = K.ones_like(K.reshape(inputs[:, 0], (-1, 1)))
        return K.tile(ones, (1, dims))
    else:
        return K.ones((K.shape(inputs)[0], dims))


def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return K.in_train_phase(
        dropped_inputs,
        ones,
        training=training)


class GRUAttentionCell(GRUCell):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.state_size = self.units :for RNN, GRU
        #self.state_size = (self.units, self.units) : for LSTM
        #self.attn_length = 79
        #self.attn_size = 79
        #self.attn_vec_size = self.attn_size
        #self.input_size = None
        self.state_size = (self.units, 79*self.units)
        #self.state_size = (self.units, self.units)
        self.full_inputs = None


    def build(self, input_shape):
        print('cell.build Shape:', input_shape)
        #self.timesteps = input_shape[1]
        input_dim = input_shape[-1]
        self.states = [None, None]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.context_kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name='context_kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_z = self.kernel[:,               : self.units * 1]
        self.kernel_r = self.kernel[:, self.units * 1: self.units * 2]
        self.kernel_h = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_c = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_z = self.recurrent_kernel[:,               : self.units * 1]
        self.recurrent_kernel_r = self.recurrent_kernel[:, self.units * 1: self.units * 2]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 3:]

        self.context_kernel_z = self.context_kernel[:,               : self.units * 1]
        self.context_kernel_r = self.context_kernel[:, self.units * 1: self.units * 2]
        self.context_kernel_h = self.context_kernel[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[              : self.units * 1]
            self.bias_r = self.bias[self.units * 1: self.units * 2]
            self.bias_h = self.bias[self.units * 2: self.units * 3]
            self.bias_c = self.bias[self.units * 3:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None
            self.bias_c = None
        self.built = True

        """
        # Parameter Shapes
        kernel : (input_dim, units)
        recurrent_kernel : (units, units)
        bias : (units, )

        # Inherited Parameters
        kernel : z, r, h
        recurrent_kernel : z, r, h
        bias : z, r, h

        # New Parameters
        input_dim
        kernel_c
        recurrent_kernel_c
        bias_c
        """

    def call(self, inputs, states, training=None):

        h_tm1 = states[0]  # previous memory
        if self.full_inputs is None:
            self.full_inputs = states[-1]
        full_inputs = self.full_inputs
        timesteps = K.int_shape(full_inputs)[1]
        print('cell.call Input Shape:', K.int_shape(inputs),
              'Full State len:', len(states),
              'State Shape:', K.int_shape(states[0]),
              'Full h Shape:', K.int_shape(states[-1]))

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
                inputs_c = inputs * dp_mask[3]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs
                inputs_c = inputs
            x_z = K.dot(inputs_z, self.kernel_z)
            x_r = K.dot(inputs_r, self.kernel_r)
            x_h = K.dot(inputs_h, self.kernel_h)
            x_c = K.dot(inputs_c, self.kernel_c)
            if self.use_bias:
                x_z = K.bias_add(x_z, self.bias_z)
                x_r = K.bias_add(x_r, self.bias_r)
                x_h = K.bias_add(x_h, self.bias_h)
                x_c = K.bias_add(x_c, self.bias_c)

            if 0. < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
                h_tm1_c = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1
                h_tm1_c = h_tm1

            # calculate the context vector
            #context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)

            # Attention Context Part
            h_tm1_c = K.repeat(h_tm1_c, timesteps)
            #h_tm1_c = full_inputs
            print('h:', K.int_shape(full_inputs), 's0:', K.int_shape(h_tm1_c))
            e = self.activation(h_tm1_c + K.dot(full_inputs, self.recurrent_kernel_c))
            a = K.softmax(e)
            print('A:', K.int_shape(a), 'inputs_c:', K.int_shape(inputs_c), 'full input:', K.int_shape(full_inputs))
            c_t = K.sum(a * full_inputs, axis=1, keepdims=False)
            #c_t = K.batch_dot(a, inputs_c, axes=1)

            print('Done c_t:', K.int_shape(c_t))
            print('Done context_kernel:', K.int_shape(self.context_kernel_z))
            print('Done recurrent_kernel:', self.recurrent_kernel_z)
            print('Done h_tm_z * rec:', K.dot(h_tm1_z, self.recurrent_kernel_z))

            # GRU Part
            z = self.recurrent_activation(x_z +
                                          K.dot(h_tm1_z,
                                                self.recurrent_kernel_z) +
                                          K.dot(c_t, self.context_kernel_z))
            r = self.recurrent_activation(x_r +
                                          K.dot(h_tm1_r,
                                                self.recurrent_kernel_r) +
                                          K.dot(c_t, self.context_kernel_r))

            hh = self.activation(x_h +
                                 K.dot(r * h_tm1_h,
                                       self.recurrent_kernel_h) +
                                 K.dot(c_t, self.context_kernel_h))

        else:
            """
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            matrix_x = K.dot(inputs, self.kernel)
            if self.use_bias:
                matrix_x = K.bias_add(matrix_x, self.bias)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            matrix_inner = K.dot(h_tm1,
                                 self.recurrent_kernel[:, :2 * self.units])

            x_z = matrix_x[:, :self.units]
            x_r = matrix_x[:, self.units: 2 * self.units]
            recurrent_z = matrix_inner[:, :self.units]
            recurrent_r = matrix_inner[:, self.units: 2 * self.units]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            x_h = matrix_x[:, 2 * self.units:]
            recurrent_h = K.dot(r * h_tm1,
                                self.recurrent_kernel[:, 2 * self.units:])
            hh = self.activation(x_h + recurrent_h)
            """
            pass

        h = z * h_tm1 + (1 - z) * hh
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        print(K.int_shape(states[-1]))
        states = [h] + [states[-1]]
        return h, [h, self.full_inputs]


class MyRNNAttention(RNN):

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        if hasattr(self.cell.state_size, '__len__'):
            state = [K.tile(initial_state, [1, dim])
                     for dim in self.cell.state_size[:-1]]
        else:
            state = [K.tile(initial_state, [1, self.cell.state_size])]

        full_inputs = [inputs]
        return state + full_inputs
