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
#from keras.layers.recurrent import (_generate_dropout_ones,
#                                    _generate_dropout_mask)

# %% ATTRNNCell ---------------------------------------------------------------

class ATTRNNCell(Layer):

    def __init__(self, units, attn_size, attn_length,
                 activation='tanh',
                 **kwargs):
        super(ATTRNNCell, self).__init__(**kwargs)
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


aa = Input(shape=(3, 1), dtype='float32')
bb = ATTRNNCell(2, 3, 3)
cc = RNN(bb, return_sequences=True, return_state=True)(aa)
#cc = LSTM(2, return_sequences=True, return_state=True)(aa)
dd = Model(inputs=aa, outputs=cc)
dd.summary()


# %% ATTRNN Layer -------------------------------------------------------------

class ATTRNN(RNN):

    def __init__(self, cell,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        super(ATTRNN, self).__init__(cell, **kwargs)
        #self.cell = cell
        #self.return_sequences = return_sequences
        #self.return_state = return_state
        #self.go_backwards = go_backwards
        #self.stateful = stateful
        #self.unroll = unroll

        #self.supports_masking = True
        #self.input_spec = [InputSpec(ndim=3)]
        #self.state_spec = None
        #self._states = None
        #self.constants_spec = None
        #self._num_constants = None

    def build(self, input_shape):
        # Note input_shape will be list of shapes of initial states and
        # constants if these are passed in __call__.
        if self._num_constants is not None:
            constants_shape = input_shape[-self._num_constants:]
        else:
            constants_shape = None

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        input_dim = input_shape[-1]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, input_dim))

        # allow cell (if layer) to build before we set or validate state_spec
        if isinstance(self.cell, Layer):
            step_input_shape = (input_shape[0],) + input_shape[2:]
            if constants_shape is not None:
                self.cell.build([step_input_shape] + constants_shape)
            else:
                self.cell.build(step_input_shape)

        # set or validate state_spec
        if hasattr(self.cell.state_size, '__len__'):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if not [spec.shape[-1] for spec in self.state_spec] == state_size:
                raise ValueError(
                    'An initial_state was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'However `cell.state_size` is '
                    '{}'.format(self.state_spec, self.cell.state_size))
        else:
            self.state_spec = [InputSpec(shape=(None, dim))
                               for dim in state_size]
        if self.stateful:
            self.reset_states()

    def get_initial_state(self, inputs):
        return super(ATTRNN, self).get_initial_state(inputs)

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = self._standardize_args(
            inputs, initial_state, constants)

        if initial_state is None and constants is None:
            return super(RNN, self).__call__(inputs, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = [InputSpec(shape=K.int_shape(state))
                               for state in initial_state]
            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        # at this point additional_inputs cannot be empty
        is_keras_tensor = hasattr(additional_inputs[0], '_keras_history')
        for tensor in additional_inputs:
            if hasattr(tensor, '_keras_history') != is_keras_tensor:
                raise ValueError('The initial state or constants of an RNN'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors')

        if is_keras_tensor:
            # Compute the full input spec, including state and constants
            full_input = [inputs] + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(RNN, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        else:
            return super(RNN, self).__call__(inputs, **kwargs)

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            inputs = inputs[0]
        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        timesteps = input_shape[1]
        if self.unroll and timesteps in [None, 1]:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined or equal to 1. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')

        kwargs = {}
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        if constants:
            if not has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')

            def step(inputs, states):
                constants = states[-self._num_constants:]
                states = states[:-self._num_constants]
                return self.cell.call(inputs, states, constants=constants,
                                      **kwargs)
        else:
            def step(inputs, states):
                return self.cell.call(inputs, states, **kwargs)

        last_output, outputs, states = fullrnn(step,
                                             inputs,
                                             initial_state,
                                             constants=constants,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             unroll=self.unroll,
                                             input_length=timesteps)
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return [output] + states
        else:
            return output

# %% RUN ----------------------------------------------------------------------

aa = Input(shape=(3, 1), dtype='float32')
bb = ATTRNNCell(2, 3, 3)
cc = ATTRNN(bb, return_sequences=True, return_state=True)(aa)
#cc = LSTM(2, return_sequences=True, return_state=True)(aa)
dd = Model(inputs=aa, outputs=cc)
dd.summary()
