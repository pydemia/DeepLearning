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

from keras import Model
#from keras.layers.recurrent import (_generate_dropout_ones,
#                                    _generate_dropout_mask)


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
                                                 training=training)

        new_inputs = K.dot(self.kk, new_inputs)
        new_inputs = K.bias_add(new_inputs, self.bb)
        return new_inputs, new_states


# %% Run ----------------------------------------------------------------------

aa = Input(shape=(3, 2), dtype='float32')
bb = SimpleRNNCell(2)
cc = CellWrapper(bb)
dd = RNN(cc, return_sequences=True, return_state=True)(aa)
#cc = LSTM(2, return_sequences=True, return_state=True)(aa)
ee = Model(inputs=aa, outputs=dd)
ee.summary()


bb.weights
bb.get_weights()

cc.weights
cc.get_weights()

cc.trainable
cc.trainable_weights
cc.count_params()

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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
fitted = model.fit(train_X, train_Y,
                   epochs=10,     # How many times to run back_propagation
                   batch_size=2,  # How many data to deal with at one epoch
                   validation_split=0.2,
                   verbose=2,       # 1: progress bar, 2: one line per epoch
                   #validation_data=(testX, testY),  # Validation set
                   shuffle=True,
                   #callbacks=[history],
                  )

# Save model
model.save('gru_attention_embedding_model.h5')
