# Control Flow


## `tf.TensorArray` & `tf.while_loop`

```py
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

with tf.variable_scope('test'):
    a = tf.get_variable(shape=(64, 8, 8, 3), name='a')
    b = tf.get_variable(shape=(64, 8, 8, 3), name='b')

batch_size = tf.shape(a)[0]
max_iteration = tf.shape(a)[0]
num_i = 0

outputs = tf.TensorArray(
    size=batch_size,
    dynamic_size=True,
    dtype=tf.int32,
)
input_ta = tf.TensorArray(
    size=batch_size,
    dtype=tf.int32,
)
input_ta = input_ta.unstack(a)

def cond_fn(num_i, max_iteration, res):
    return num_i < max_iteration

def body_fn(num_i, max_iteration, output_ta_t):

    xt = input_ta.read(num_i)
    new_output = tf.reduce_mean(xt)
    output_ta_t = output_ta_t.write(num_i, new_output)

    next_i = num_i + 1
    return next_i, max_iteration, output_ta_t

_, _, result = tf.while_loop(
    cond_fn,
    body_fn,
    loop_vars=[num_i, max_iteration, outputs],
    shape_invariants=None,
)

stacked = result.stack()

print(stacked.shape)
```


## `tf.gather`(Indexing & Slicing)

```py

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

with tf.variable_scope('test'):
    a = tf.get_variable(shape=(64, 8, 8, 3), dtype=tf.float32, name='a')
    idx = tf.convert_to_tensor(np.array([1, 2, 1, 4, 2, 4, 5]), dtype=tf.int32)

    # selected = tf.gather_nd(
    #     a,
    #     idx,
    # )
    selected = tf.gather(
        a,
        idx,
    )

    print(selected.shape)
```
