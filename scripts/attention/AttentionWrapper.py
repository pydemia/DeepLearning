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

    def __init__(self, **kwargs):
        super(SkeletonCell, self).__init__(**kwargs)

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


# %% Attention(Cell) Wrapper --------------------------------------------------------

class CellWrapper(SimpleRNNCell):
    def __init__(self, cell, *args, **kwargs):
        super(CellWrapper, self).__init__(**kwargs)
        self._cell = cell
        #self._cell.__init__(*args, **kwargs)

    @property
    def units(self):
        return self._cell.units

    @property
    def activation(self):
        return self._cell.activation

    @property
    def use_bias(self):
        return self._cell.use_bias

    @property
    def kernel_initializer(self):
        return self._cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self._cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self._cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self._cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self._cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self._cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self._cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self._cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self._cell.bias_constraint

    @property
    def dropout(self):
        return self._cell.dropout

    @property
    def recurrent_dropout(self):
        return self._cell.recurrent_dropout

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def implementation(self):
        return self.cell.implementation

    def build(self, input_shape):
        self._cell.build(input_shape)
        if self.built is None:
            self.built = True

    def call(self, inputs, states, training=None):

        cell_output, new_state = self._cell.call(inputs, state, training=None)

        output = cell_output
        state = new_state

        return output, state


class AttentionCellWrapper(CellWrapper):

    def __init__(self, cell, *args, **kwargs):
        super(AttentionCellWrapper, self).__init__(*args, **kwargs):

        pass

    def build(self, input_shape):
        super(AttentionCellWrapper, self).build(input_shape)
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
