class AttentionCellWrapper(rnn_cell_impl.RNNCell):
    """Basic attention cell wrapper.
    Implementation based on https://arxiv.org/abs/1409.0473.
    """

    def __init__(self, cell, attn_length, attn_size=None, attn_vec_size=None,
                             input_size=None, state_is_tuple=True, reuse=None):
        """Create a cell with attention.
        Args:
            cell: an RNNCell, an attention is added to it.
            attn_length: integer, the size of an attention window.
            attn_size: integer, the size of an attention vector. Equal to
                    cell.output_size by default.
            attn_vec_size: integer, the number of convolutional features calculated
                    on attention state and a size of the hidden layer built from
                    base cell state. Equal attn_size to by default.
            input_size: integer, the size of a hidden linear layer,
                    built from inputs and attention. Derived from the input tensor
                    by default.
            state_is_tuple: If True, accepted and returned states are n-tuples, where
                `n = len(cells)`.    By default (False), the states are all
                concatenated along the column axis.
            reuse: (optional) Python boolean describing whether to reuse variables
                in an existing scope.    If not `True`, and the existing scope already has
                the given variables, an error is raised.
        Raises:
            TypeError: if cell is not an RNNCell.
            ValueError: if cell returns a state tuple but the flag
                    `state_is_tuple` is `False` or if attn_length is zero or less.
        """
        super(AttentionCellWrapper, self).__init__(_reuse=reuse)
        if not rnn_cell_impl._like_rnncell(cell):    # pylint: disable=protected-access
            raise TypeError("The parameter cell is not RNNCell.")
        if nest.is_sequence(cell.state_size) and not state_is_tuple:
            raise ValueError("Cell returns tuple of states, but the flag "
                                             "state_is_tuple is not set. State size is: %s"
                                             % str(cell.state_size))
        if attn_length <= 0:
            raise ValueError("attn_length should be greater than zero, got %s"
                                             % str(attn_length))
        if not state_is_tuple:
            logging.warn(
                    "%s: Using a concatenated state is slower and will soon be "
                    "deprecated.    Use state_is_tuple=True.", self)
        if attn_size is None:
            attn_size = cell.output_size
        if attn_vec_size is None:
            attn_vec_size = attn_size
        self._state_is_tuple = state_is_tuple
        self._cell = cell
        self._attn_vec_size = attn_vec_size
        self._input_size = input_size
        self._attn_size = attn_size
        self._attn_length = attn_length
        self._reuse = reuse
        self._linear1 = None
        self._linear2 = None
        self._linear3 = None

    @property
    def state_size(self):
        size = (self._cell.state_size, self._attn_size,
                        self._attn_size * self._attn_length)
        if self._state_is_tuple:
            return size
        else:
            return sum(list(size))

    @property
    def output_size(self):
        return self._attn_size

    def call(self, inputs, state):
        """Long short-term memory cell with attention (LSTMA)."""
        if self._state_is_tuple:
            state, attns, attn_states = state
        else:
            states = state
            state = array_ops.slice(states, [0, 0], [-1, self._cell.state_size])
            attns = array_ops.slice(
                    states, [0, self._cell.state_size], [-1, self._attn_size])
            attn_states = array_ops.slice(
                    states, [0, self._cell.state_size + self._attn_size],
                    [-1, self._attn_size * self._attn_length])
        attn_states = array_ops.reshape(attn_states, [-1, self._attn_length, self._attn_size])
        input_size = self._input_size
        if input_size is None:
            input_size = inputs.get_shape().as_list()[1]
        if self._linear1 is None:
            self._linear1 = _Linear([inputs, attns], input_size, True)
        inputs = self._linear1([inputs, attns])
        cell_output, new_state = self._cell(inputs, state)
        if self._state_is_tuple:
            new_state_cat = array_ops.concat(nest.flatten(new_state), 1)
        else:
            new_state_cat = new_state

        new_attns, new_attn_states = self._attention(new_state_cat, attn_states)
        with vs.variable_scope("attn_output_projection"):
            if self._linear2 is None:
                self._linear2 = _Linear([cell_output, new_attns], self._attn_size, True)
            output = self._linear2([cell_output, new_attns])
        new_attn_states = array_ops.concat(
                [new_attn_states, array_ops.expand_dims(output, 1)], 1)
        new_attn_states = array_ops.reshape(
                new_attn_states, [-1, self._attn_length * self._attn_size])
        new_state = (new_state, new_attns, new_attn_states)
        if not self._state_is_tuple:
            new_state = array_ops.concat(list(new_state), 1)
        return output, new_state

    def _attention(self, query, attn_states):
        conv2d = nn_ops.conv2d
        reduce_sum = math_ops.reduce_sum
        softmax = nn_ops.softmax
        tanh = math_ops.tanh

        with vs.variable_scope("attention"):
            k = vs.get_variable(
                    "attn_w", [1, 1, self._attn_size, self._attn_vec_size])
            v = vs.get_variable("attn_v", [self._attn_vec_size])
            hidden = array_ops.reshape(attn_states, [-1, self._attn_length, 1, self._attn_size])
            hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            if self._linear3 is None:
                self._linear3 = _Linear(query, self._attn_vec_size, True)
            y = self._linear3(query)
            y = array_ops.reshape(y, [-1, 1, 1, self._attn_vec_size])
            s = reduce_sum(v * tanh(hidden_features + y), [2, 3])
            a = softmax(s)
            d = reduce_sum(array_ops.reshape(a, [-1, self._attn_length, 1, 1]) * hidden, [1, 2])
            new_attns = array_ops.reshape(d, [-1, self._attn_size])
            new_attn_states = array_ops.slice(attn_states, [0, 1, 0], [-1, -1, -1])
return new_attns, new_attn_states
