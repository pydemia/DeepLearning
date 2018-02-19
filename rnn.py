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

# Legacy support
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


        class SimpleRNNCell(Layer):
            """Cell class for SimpleRNN.
            """

            def __init__(self, units,
                         activation='tanh',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         kernel_regularizer=None,
                         recurrent_regularizer=None,
                         bias_regularizer=None,
                         kernel_constraint=None,
                         recurrent_constraint=None,
                         bias_constraint=None,
                         dropout=0.,
                         recurrent_dropout=0.,
                         **kwargs):
                super(SimpleRNNCell, self).__init__(**kwargs)
                self.units = units
                self.activation = activations.get(activation)
                self.use_bias = use_bias

                self.kernel_initializer = initializers.get(kernel_initializer)
                self.recurrent_initializer = initializers.get(recurrent_initializer)
                self.bias_initializer = initializers.get(bias_initializer)

                self.kernel_regularizer = regularizers.get(kernel_regularizer)
                self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
                self.bias_regularizer = regularizers.get(bias_regularizer)

                self.kernel_constraint = constraints.get(kernel_constraint)
                self.recurrent_constraint = constraints.get(recurrent_constraint)
                self.bias_constraint = constraints.get(bias_constraint)

                self.dropout = min(1., max(0., dropout))
                self.recurrent_dropout = min(1., max(0., recurrent_dropout))
                self.state_size = self.units
                self._dropout_mask = None
                self._recurrent_dropout_mask = None

            def build(self, input_shape):
                self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                              name='kernel',
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)
                self.recurrent_kernel = self.add_weight(
                    shape=(self.units, self.units),
                    name='recurrent_kernel',
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)
                if self.use_bias:
                    self.bias = self.add_weight(shape=(self.units,),
                                                name='bias',
                                                initializer=self.bias_initializer,
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint)
                else:
                    self.bias = None
                self.built = True

            def call(self, inputs, states, training=None):
                prev_output = states[0]
                if 0 < self.dropout < 1 and self._dropout_mask is None:
                    self._dropout_mask = _generate_dropout_mask(
                        _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                        self.dropout,
                        training=training)
                if (0 < self.recurrent_dropout < 1 and
                        self._recurrent_dropout_mask is None):
                    self._recurrent_dropout_mask = _generate_dropout_mask(
                        _generate_dropout_ones(inputs, self.units),
                        self.recurrent_dropout,
                        training=training)

                dp_mask = self._dropout_mask
                rec_dp_mask = self._recurrent_dropout_mask

                if dp_mask is not None:
                    h = K.dot(inputs * dp_mask, self.kernel)
                else:
                    h = K.dot(inputs, self.kernel)
                if self.bias is not None:
                    h = K.bias_add(h, self.bias)

                if rec_dp_mask is not None:
                    prev_output *= rec_dp_mask
                output = h + K.dot(prev_output, self.recurrent_kernel)
                if self.activation is not None:
                    output = self.activation(output)

                # Properly set learning phase on output tensor.
                if 0 < self.dropout + self.recurrent_dropout:
                    if training is None:
                        output._uses_learning_phase = True
                return output, [output]


        class SimpleRNN(RNN):
            """Fully-connected RNN where the output is to be fed back to input.
            """

            @interfaces.legacy_recurrent_support
            def __init__(self, units,
                         activation='tanh',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         kernel_regularizer=None,
                         recurrent_regularizer=None,
                         bias_regularizer=None,
                         activity_regularizer=None,
                         kernel_constraint=None,
                         recurrent_constraint=None,
                         bias_constraint=None,
                         dropout=0.,
                         recurrent_dropout=0.,
                         return_sequences=False,
                         return_state=False,
                         go_backwards=False,
                         stateful=False,
                         unroll=False,
                         **kwargs):
                if 'implementation' in kwargs:
                    kwargs.pop('implementation')
                    warnings.warn('The `implementation` argument '
                                  'in `SimpleRNN` has been deprecated. '
                                  'Please remove it from your layer call.')
                if K.backend() == 'theano':
                    warnings.warn(
                        'RNN dropout is no longer supported with the Theano backend '
                        'due to technical limitations. '
                        'You can either set `dropout` and `recurrent_dropout` to 0, '
                        'or use the TensorFlow backend.')
                    dropout = 0.
                    recurrent_dropout = 0.

                cell = SimpleRNNCell(units,
                                     activation=activation,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     recurrent_initializer=recurrent_initializer,
                                     bias_initializer=bias_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     recurrent_regularizer=recurrent_regularizer,
                                     bias_regularizer=bias_regularizer,
                                     kernel_constraint=kernel_constraint,
                                     recurrent_constraint=recurrent_constraint,
                                     bias_constraint=bias_constraint,
                                     dropout=dropout,
                                     recurrent_dropout=recurrent_dropout)
                super(SimpleRNN, self).__init__(cell,
                                                return_sequences=return_sequences,
                                                return_state=return_state,
                                                go_backwards=go_backwards,
                                                stateful=stateful,
                                                unroll=unroll,
                                                **kwargs)
                self.activity_regularizer = regularizers.get(activity_regularizer)

            def call(self, inputs, mask=None, training=None, initial_state=None):
                return super(SimpleRNN, self).call(inputs,
                                                   mask=mask,
                                                   training=training,
                                                   initial_state=initial_state)

            @property
            def units(self):
                return self.cell.units

            @property
            def activation(self):
                return self.cell.activation

            @property
            def use_bias(self):
                return self.cell.use_bias

            @property
            def kernel_initializer(self):
                return self.cell.kernel_initializer

            @property
            def recurrent_initializer(self):
                return self.cell.recurrent_initializer

            @property
            def bias_initializer(self):
                return self.cell.bias_initializer

            @property
            def kernel_regularizer(self):
                return self.cell.kernel_regularizer

            @property
            def recurrent_regularizer(self):
                return self.cell.recurrent_regularizer

            @property
            def bias_regularizer(self):
                return self.cell.bias_regularizer

            @property
            def kernel_constraint(self):
                return self.cell.kernel_constraint

            @property
            def recurrent_constraint(self):
                return self.cell.recurrent_constraint

            @property
            def bias_constraint(self):
                return self.cell.bias_constraint

            @property
            def dropout(self):
                return self.cell.dropout

            @property
            def recurrent_dropout(self):
                return self.cell.recurrent_dropout

            def get_config(self):
                config = {'units': self.units,
                          'activation': activations.serialize(self.activation),
                          'use_bias': self.use_bias,
                          'kernel_initializer': initializers.serialize(self.kernel_initializer),
                          'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                          'bias_initializer': initializers.serialize(self.bias_initializer),
                          'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                          'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                          'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                          'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                          'kernel_constraint': constraints.serialize(self.kernel_constraint),
                          'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                          'bias_constraint': constraints.serialize(self.bias_constraint),
                          'dropout': self.dropout,
                          'recurrent_dropout': self.recurrent_dropout}
                base_config = super(SimpleRNN, self).get_config()
                del base_config['cell']
                return dict(list(base_config.items()) + list(config.items()))

            @classmethod
            def from_config(cls, config):
                if 'implementation' in config:
                    config.pop('implementation')
                return cls(**config)
