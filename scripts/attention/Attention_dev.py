# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import warnings
from pprint import pprint

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

from keras import Model
from keras.callbacks import Callback, LambdaCallback

#from keras.layers.recurrent import (_generate_dropout_ones,
#                                    _generate_dropout_mask)


# %% CallBack -----------------------------------------------------------------

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.weights = []
        self.states = []

    def on_batch_begin(self, batch, logs={}):
        self.weights.append([{'begin_' + layer.name: layer.get_weights()}
                             for layer in model.layers])

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.weights.append([{'end_' + layer.name: layer.get_weights()}
                             for layer in model.layers])


history = LossHistory()

print_weights = LambdaCallback(on_epoch_end=lambda batch,
                               logs: pprint(model.layers[0].get_weights()))

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


class CellWrapper(Layer):

    """
    A Cell Wrapper for Attention Mechanism.
    """

    def __init__(self, cell, *args, **kwargs):
        super(CellWrapper, self).__init__(*args, **kwargs)
        self._cell = cell
        self._cell.__init__(self.units, **kwargs)

        self.trainable = self._cell.trainable

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

#    @property
#    def weights(self):
#        return self._cell.weights + super(CellWrapper, self).weights

#    @property
#    def get_weights(self):
#        #return self._cell.weights + self.weights
#        #return self.weights
#
#        params = self.weights
#        return K.batch_get_value(params)

    def build(self, input_shape):

        self._cell.build(input_shape)
        if self._cell.trainable:
            self._trainable_weights.extend(self._cell.weights)
        else:
            self._non_trainable_weights.extend(self._cell.weights)

        input_dim = input_shape[-1]
        self.kk = self.add_weight(shape=(input_dim, self.units,),
                                  name='wrapper_kernel',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)

        self.bb = self.add_weight(shape=(self.units,),
                                  name='wrapper_bias',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        """
        self.kernel
        self.recurrent_kernel

        """
        self.built = True

    def call(self, inputs, states, training=None, constants=None):
        new_inputs, new_states = self._cell.call(inputs, states,
                                                 training=training,
                                                 constants=constants)

        new_inputs = K.dot(self.kk, new_inputs)
        new_inputs = K.bias_add(new_inputs, self.bb)
        return new_inputs, new_states


# %% Run ----------------------------------------------------------------------

aa = Input(shape=(3, 2), dtype='float32')
bb = SimpleRNNCell(2)
cc = CellWrapper(bb)
dd = RNN(cc, return_sequences=True, return_state=False)(aa)
ee = Model(inputs=aa, outputs=dd)
ee.summary()


bb.weights
bb.get_weights()

cc.weights
cc.get_weights()

cc.trainable
cc.trainable_weights
cc.count_params()

ee.get_weights()
# %% Learning -----------------------------------------------------------------

import numpy as np

train_X = np.random.sample(300).reshape(-1, 3, 2)
train_X

train_Y = np.random.sample(300).reshape(-1, 3, 2)
train_Y


EPOCH_NUM = 10
BATCH_SIZE = 3
print('EPOCH_NUM: %s, BATCH_SIZE %s' % (EPOCH_NUM, BATCH_SIZE))

model = ee
model.compile(optimizer='adam', loss='mae', metrics=['mae'])
fitted = model.fit(train_X, train_Y,
                   epochs=10,     # How many times to run back_propagation
                   batch_size=2,  # How many data to deal with at one epoch
                   validation_split=0.2,
                   verbose=2,       # 1: progress bar, 2: one line per epoch
                   #validation_data=(testX, testY),  # Validation set
                   shuffle=True,
                   callbacks=[history],
                  )

# Save model
model.save('cell_wrapper_model.h5')


fitted.history

history.weights[:4]

ee.get_weights()


# %% RNN Wrapper --------------------------------------------------------------

class RNNWrapper(RNN):


    def __init__(self, RNN,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if isinstance(cell, (list, tuple)):
            cell = StackedRNNCells(cell)
        if not hasattr(cell, 'call'):
            raise ValueError('`cell` should have a `call` method. '
                             'The RNN was passed:', cell)
        if not hasattr(cell, 'state_size'):
            raise ValueError('The RNN cell should have '
                             'an attribute `state_size` '
                             '(tuple of integers, '
                             'one integer per RNN state).')
        super(RNNWrapper, self).__init__(**kwargs)
        self.cell = RNN.cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll

        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.state_spec = None
        self._states = None
        self.constants_spec = None
        self._num_constants = None

    @property
    def states(self):
        if self._states is None:
            if isinstance(self.cell.state_size, int):
                num_states = 1
            else:
                num_states = len(self.cell.state_size)
            return [None for _ in range(num_states)]
        return self._states

    @states.setter
    def states(self, states):
        self._states = states

    def compute_output_shape(self, input_shape):
        return super(RNNWrapper, self).compute_output_shape(input_shape)

    def compute_mask(self, inputs, mask):
        return super(RNNWrapper, self).compute_mask(inputs, mask)

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
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        if hasattr(self.cell.state_size, '__len__'):
            return [K.tile(initial_state, [1, dim])
                    for dim in self.cell.state_size]
        else:
            return [K.tile(initial_state, [1, self.cell.state_size])]

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = self._standardize_args(
            inputs, initial_state, constants)

        if initial_state is None and constants is None:
            return super(RNNWrapper, self).__call__(inputs, **kwargs)

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
            output = super(RNNWrapper, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        else:
            return super(RNNWrapper, self).__call__(inputs, **kwargs)

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

        #constants = K.stack(inputs)
        constants = [inputs]
        last_output, outputs, states = K.rnn(step,
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

    def _standardize_args(self, inputs, initial_state, constants):
        """Brings the arguments of `__call__` that can contain input tensors to
        standard format.

        When running a model loaded from file, the input tensors
        `initial_state` and `constants` can be passed to `RNN.__call__` as part
        of `inputs` instead of by the dedicated keyword arguments. This method
        makes sure the arguments are separated and that `initial_state` and
        `constants` are lists of tensors (or None).

        # Arguments
            inputs: tensor or list/tuple of tensors
            initial_state: tensor or list of tensors or None
            constants: tensor or list of tensors or None

        # Returns
            inputs: tensor
            initial_state: list of tensors or None
            constants: list of tensors or None
        """
        if isinstance(inputs, list):
            assert initial_state is None and constants is None
            if self._num_constants is not None:
                constants = inputs[-self._num_constants:]
                inputs = inputs[:-self._num_constants]
            if len(inputs) > 1:
                initial_state = inputs[1:]
            inputs = inputs[0]

        def to_list_or_none(x):
            if x is None or isinstance(x, list):
                return x
            if isinstance(x, tuple):
                return list(x)
            return [x]

        initial_state = to_list_or_none(initial_state)
        constants = to_list_or_none(constants)

        return inputs, initial_state, constants

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        batch_size = self.input_spec[0].shape[0]
        if not batch_size:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')
        # initialize state if None
        if self.states[0] is None:
            if hasattr(self.cell.state_size, '__len__'):
                self.states = [K.zeros((batch_size, dim))
                               for dim in self.cell.state_size]
            else:
                self.states = [K.zeros((batch_size, self.cell.state_size))]
        elif states is None:
            if hasattr(self.cell.state_size, '__len__'):
                for state, dim in zip(self.states, self.cell.state_size):
                    K.set_value(state, np.zeros((batch_size, dim)))
            else:
                K.set_value(self.states[0],
                            np.zeros((batch_size, self.cell.state_size)))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 ' state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if hasattr(self.cell.state_size, '__len__'):
                    dim = self.cell.state_size[index]
                else:
                    dim = self.cell.state_size
                if value.shape != (batch_size, dim):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str((batch_size, dim)) +
                                     ', found shape=' + str(value.shape))
                # TODO: consider batch calls to `set_value`.
                K.set_value(state, value)

    def get_config(self):
        config = {'return_sequences': self.return_sequences,
                  'return_state': self.return_state,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll}
        if self._num_constants is not None:
            config['num_constants'] = self._num_constants

        cell_config = self.cell.get_config()
        config['cell'] = {'class_name': self.cell.__class__.__name__,
                          'config': cell_config}
        base_config = super(RNNWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from . import deserialize as deserialize_layer
        cell = deserialize_layer(config.pop('cell'),
                                 custom_objects=custom_objects)
        num_constants = config.pop('num_constants', None)
        layer = cls(cell, **config)
        layer._num_constants = num_constants
        return layer

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        if isinstance(self.cell, Layer):
            return self.cell.trainable_weights
        return []

    @property
    def non_trainable_weights(self):
        if isinstance(self.cell, Layer):
            if not self.trainable:
                return self.cell.weights
            return self.cell.non_trainable_weights
        return []

    @property
    def losses(self):
        if isinstance(self.cell, Layer):
            return self.cell.losses
        return []

    def get_losses_for(self, inputs=None):
        if isinstance(self.cell, Layer):
            cell_losses = self.cell.get_losses_for(inputs)
            return cell_losses + super(RNNWrapper, self).get_losses_for(inputs)
        return super(RNNWrapper, self).get_losses_for(inputs)
