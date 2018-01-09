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


# %% Skeleton -----------------------------------------------------------------

class SkeletonCell(Layer):

    def __init__(self, *args, **kwargs):
        pass

    def build(self, input_shape):
        pass

    def call(self, inputs, states, training=None):
        pass


class Skeleton(RNN):

    def __init__(self, *args, **kwargs):
        pass

    def call(self, inputs, initial_state=None,
             mask=None,
             training=None,
             constants=None):
        pass

# %% Minimal RNN Implementation -----------------------------------------------


class MinimalRNNCell(Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]


"""
# Let's use this cell in a RNN layer:

cell = MinimalRNNCell(32)
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)

# Here's how to use the cell to build a stacked RNN:

cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)
"""

# %% Wrapper ------------------------------------------------------------------


class Wrapper(Layer):
    """Abstract wrapper base class.

    Wrappers take another layer and augment it in various ways.
    Do not use this class as a layer, it is only an abstract base class.
    Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.

    # Arguments
        layer: The layer to be wrapped.
    """

    def __init__(self, layer, **kwargs):
        self.layer = layer
        # Tracks mapping of Wrapper inputs to inner layer inputs. Useful when
        # the inner layer has update ops that depend on its inputs (as opposed
        # to the inputs to the Wrapper layer).
        self._input_map = {}
        super(Wrapper, self).__init__(**kwargs)

    def build(self, input_shape=None):
        self.built = True

    @property
    def activity_regularizer(self):
        if hasattr(self.layer, 'activity_regularizer'):
            return self.layer.activity_regularizer
        else:
            return None

    @property
    def trainable_weights(self):
        return self.layer.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.layer.non_trainable_weights

    @property
    def updates(self):
        if hasattr(self.layer, 'updates'):
            return self.layer.updates
        return []

    def get_updates_for(self, inputs=None):
        # If the wrapper modifies the inputs, use the modified inputs to
        # get the updates from the inner layer.
        inner_inputs = inputs
        if inputs is not None:
            uid = _object_list_uid(inputs)
            if uid in self._input_map:
                inner_inputs = self._input_map[uid]

        updates = self.layer.get_updates_for(inner_inputs)
        updates += super(Wrapper, self).get_updates_for(inputs)
        return updates

    @property
    def losses(self):
        if hasattr(self.layer, 'losses'):
            return self.layer.losses
        return []

    def get_losses_for(self, inputs=None):
        if inputs is None:
            losses = self.layer.get_losses_for(None)
            return losses + super(Wrapper, self).get_losses_for(None)
        return super(Wrapper, self).get_losses_for(inputs)

    def get_weights(self):
        return self.layer.get_weights()

    def set_weights(self, weights):
        self.layer.set_weights(weights)

    def get_config(self):
        config = {'layer': {'class_name': self.layer.__class__.__name__,
                            'config': self.layer.get_config()}}
        base_config = super(Wrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from . import deserialize as deserialize_layer
        layer = deserialize_layer(config.pop('layer'),
                                  custom_objects=custom_objects)
        return cls(layer, **config)


# %% Attention Layer ----------------------------------------------------------


class GRUAttentionCell(GRUCell):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        #self.timesteps = input_shape[0]
        input_dim = input_shape[-1]
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
        input_shape = K.int_shape(inputs)
        timesteps = input_shape[1]
        h_tm1 = states[0]  # previous memory

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
            e = self.activation(x_c + K.dot(h_tm1_c, self.recurrent_kernel_c))
            a = K.softmax(e)
            c_t = K.dot(a, inputs_c)

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
        h = z * h_tm1 + (1 - z) * hh
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h]



class GRUAttention(GRU):

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
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
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if K.backend() == 'theano':
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = GRUAttentionCell(units,
                                activation=activation,
                                recurrent_activation=recurrent_activation,
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
                                recurrent_dropout=recurrent_dropout,
                                implementation=implementation)
        super(GRU, self).__init__(cell,
                                  return_sequences=return_sequences,
                                  return_state=return_state,
                                  go_backwards=go_backwards,
                                  stateful=stateful,
                                  unroll=unroll,
                                  **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        #self.cell = cell
        #self.state_size = self.units
        #self._states = states

#    def get_initial_state(self, inputs):
#        pass
        """
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        if hasattr(self.cell.state_size, '__len__'):
            return [K.tile(initial_state, [1, dim])
                    for dim in self.cell.state_size]
        else:
            return [K.tile(initial_state, [1, self.cell.state_size])]
        """

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(GRU, self).call(inputs,
                                     mask=mask,
                                     training=training,
                                     initial_state=initial_state)


# %% Attention Wrapper --------------------------------------------------------


class AttentionCellWrapper(Wrapper):

    def __init__(self):
        pass

    def build(self, input_shape):
        return super().build(input_shape)
        """
        self.kernel
        self.recurrent_kernel

        """

    def call(self, inputs, states, training=None):
        pass


class AttentionWrapper(Wrapper):

    def __init__(self):
        pass

    def get_initial_state(self, inputs):
        pass
        """
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        if hasattr(self.cell.state_size, '__len__'):
            return [K.tile(initial_state, [1, dim])
                    for dim in self.cell.state_size]
        else:
            return [K.tile(initial_state, [1, self.cell.state_size])]
        """

    def call(self, inputs, initial_state=None,
             mask=None,
             training=None,
             constants=None):
        pass
