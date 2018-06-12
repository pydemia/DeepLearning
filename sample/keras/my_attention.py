from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyAttention(Recurrent):

    def __init__(self, units, output_dim,
                 activation='tanh',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 #kernel_initializer='glorot_uniform',
                 #recurrent_initializer='orthogonal',
                 #bias_initializer='zeros',
                 #kernel_regularizer=None,
                 #bias_regularizer=None,
                 #activity_regularizer=None,
                 #kernel_constraint=None,
                 #bias_constraint=None,
                 **kwargs):

        self.name = name
        self.units = units
        self.output_dim = output_dim
        self.return_sequences = True
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        #self.kernel_initializer = initializers.get(kernel_initializer)
        #self.recurrent_initializer = initializers.get(recurrent_initializer)
        #self.bias_initializer = initializers.get(bias_initializer)

        #self.kernel_regularizer = regularizers.get(kernel_regularizer)
        #self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        #self.bias_regularizer = regularizers.get(bias_regularizer)
        #self.activity_regularizer = regularizers.get(activity_regularizer)

        #self.kernel_constraint = constraints.get(kernel_constraint)
        #self.recurrent_constraint = constraints.get(kernel_constraint)
        #self.bias_constraint = constraints.get(bias_constraint)

        super(MyAttention, self).__init__(**kwargs)

    def build(self, input_shape):


        self.batch_size, self.timesteps, self.input_dim = input_shape
        #super(MyAttention, self).build(input_shape)  # Be sure to call this somewhere!

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        self.states = [None, None]  # y, s

        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        # Matrices for creating the context vector
        self.V_a = self.add_weight(shape=(self.units,), name='V_a')
        self.W_a = self.add_weight(shape=(self.units, self.units), name='W_a')
        self.U_a = self.add_weight(shape=(self.input_dim, self.units), name='U_a')
        self.b_a = self.add_weight(shape=(self.units,), name='b_a')

        # Matrices for the r (reset) gate
        self.C_r = self.add_weight(shape=(self.input_dim, self.units), name='C_r')
        self.U_r = self.add_weight(shape=(self.units, self.units), name='U_r')
        self.W_r = self.add_weight(shape=(self.output_dim, self.units), name='W_r')
        self.b_r = self.add_weight(shape=(self.units, ), name='b_r')

        # Matrices for the z (update) gate
        self.C_z = self.add_weight(shape=(self.input_dim, self.units), name='C_z')
        self.U_z = self.add_weight(shape=(self.units, self.units), name='U_z',)
        self.W_z = self.add_weight(shape=(self.output_dim, self.units), name='W_z')
        self.b_z = self.add_weight(shape=(self.units, ), name='b_z')

        # Matrices for the proposal
        self.C_p = self.add_weight(shape=(self.input_dim, self.units), name='C_p')
        self.U_p = self.add_weight(shape=(self.units, self.units), name='U_p')
        self.W_p = self.add_weight(shape=(self.output_dim, self.units), name='W_p')
        self.b_p = self.add_weight(shape=(self.units, ), name='b_p')

        # Matrices for making the final prediction vector
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim), name='C_o')
        self.U_o = self.add_weight(shape=(self.units, self.output_dim), name='U_o')
        self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim), name='W_o')
        self.b_o = self.add_weight(shape=(self.output_dim, ), name='b_o')

        # For creating the initial state:
        self.W_s = self.add_weight(shape=(self.input_dim, self.units), name='W_s')

        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True

    def call(self, x):
        # store the whole sequence so we can "attend" to it at each timestep
        self.x_seq = x

        # apply the a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:
        self._uxpb = TimeDistributed(Dense(self.x_seq, self.U_a, b=self.b_a,),
                                           input_dim=self.input_dim,
                                           timesteps=self.timesteps,
                                           output_dim=self.units)

        return super(AttentionDecoder, self).call(x)


    def get_initial_state(self, inputs):
        # apply the matrix on the first time step to get the initial s0.
        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def step(self, x, states):

        ytm, stm = states

        # repeat the hidden state to the length of the sequence
        _stm = K.repeat(stm, self.timesteps)

        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = K.dot(_stm, self.W_a)

        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.
        et = K.dot(activations.tanh(_Wxstm + self._uxpb),
                   K.expand_dims(self.V_a))
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
        # ~~~> calculate new hidden state
        # first calculate the "r" gate:

        rt = activations.sigmoid(
            K.dot(ytm, self.W_r)
            + K.dot(stm, self.U_r)
            + K.dot(context, self.C_r)
            + self.b_r)

        # now calculate the "z" gate
        zt = activations.sigmoid(
            K.dot(ytm, self.W_z)
            + K.dot(stm, self.U_z)
            + K.dot(context, self.C_z)
            + self.b_z)

        # calculate the proposal hidden state:
        s_tp = activations.tanh(
            K.dot(ytm, self.W_p)
            + K.dot((rt * stm), self.U_p)
            + K.dot(context, self.C_p)
            + self.b_p)

        # new hidden state:
        st = (1-zt)*stm + zt * s_tp

        yt = activations.softmax(
            K.dot(ytm, self.W_o)
            + K.dot(stm, self.U_o)
            + K.dot(context, self.C_o)
            + self.b_o)

        if self.return_probabilities:
            return at, [yt, st]
        else:
            return yt, [yt, st]

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
