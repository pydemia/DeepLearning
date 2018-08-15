# Optimization

## Freeze some variables

### `tf.stop_gradient()`


### `tf.GraphKeys`

```py

import tensorflow as tf
import numpy as np


input_x = np.random.normal(size=(5, 3))
input_x.shape
input_y = np.random.normal(size=(5, 4))

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(5, 3), name='input_x')
Y = tf.placeholder(tf.float32, shape=(5, 4), name='input_y')

with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):
    w_a = tf.get_variable('w_a', shape=(3, 2), dtype=tf.float32)
    b_a = tf.get_variable('b_a', shape=(2,), dtype=tf.float32)

    lineared_a = tf.nn.bias_add(X @ w_a, b_a, name='lineared_a')

with tf.variable_scope('dim', reuse=tf.AUTO_REUSE):
    w_b = tf.get_variable('w_b', shape=(2, 4), dtype=tf.float32)
    b_b = tf.get_variable('b_b', shape=(4,), dtype=tf.float32)

    lineared_b = tf.nn.bias_add(lineared_a @ w_b, b_b, name='lineared_b')

    loss = tf.losses.mean_squared_error(labels=input_y, predictions=lineared_b)

with tf.name_scope('optimization'):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=.05,
        name='optimizer',
    )
    train_op = optimizer.minimize(
        loss,
        var_list=tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='dim',
        )
    )

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    loss_list = []
    b_a_list = []
    b_b_list = []
    for _ in range(10):
        _, loss_val, b_a_val, b_b_val = sess.run(
            [train_op, loss, b_a, b_b],
            feed_dict={X: input_x, Y: input_y},
        )
        loss_list += [loss_val]
        b_a_list += [b_a_val[0]]
        b_b_list += [b_b_val[0]]


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 6))
for ax, val_list in zip(axes, [loss_list, b_a_list, b_b_list]):
    ax.plot(val_list)

```
