# Dataset


## Basic

```py

if __name__ == '__main__':
    tf.reset_default_graph()
    X_t = tf.placeholder(tf.int16, (None, 2),
                         name='x_tensor_interface')
    Z_t = tf.placeholder(tf.int16,  (None, 1),
                         name='z_tensor_interface')

    dataset = tf.data.Dataset.from_tensor_slices((X_t, Z_t))
    dataset = dataset.shuffle(buffer_size=1000)  # reshuffle_each_iteration=True as default.
    dataset = dataset.batch(2)
    dataset = dataset.flat_map(
        lambda data_x, data_z: tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensors(data_x),
                tf.data.Dataset.from_tensors(data_z),
            )
        ).repeat(12)
    )


    data_op = dataset.make_initializable_iterator()
    data_init_op = data_op.initializer
    X_batch, Z_batch = data_op.get_next()

    bias_x0 = tf.convert_to_tensor(np.array([1, 2]), dtype=tf.int16)
    bias_z0 = tf.convert_to_tensor(np.array([7]), dtype=tf.int16)

    bias_x1 = tf.convert_to_tensor(np.array([10, 11]), dtype=tf.int16)
    bias_z1 = tf.convert_to_tensor(np.array([50]), dtype=tf.int16)
    add1 = tf.nn.bias_add(X_batch, bias_x0)
    add2 = tf.nn.bias_add(Z_batch, bias_z0)
    add3 = tf.nn.bias_add(X_batch, bias_x1)
    add4 = tf.nn.bias_add(Z_batch, bias_z1)

    a = np.array([
        [100, 100],
        [200, 200],
        [300, 300],
        [400, 400],
        [500, 500],
    ])

    b = np.array([
        [600],
        [700],
        [800],
        [900],
        [600],
    ])

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        init_op.run()

        for epoch in range(10):
            print('[EPOCH]', epoch+1, '======================')
            xx, zz = sess.run(data_init_op, feed_dict={X_t: a, Z_t: b})

            print(xx)
            print(zz)



## Feed Flexible

```py
# %% Test Code: tf.data.Dataset ----------------------------------------------

if __name__ == '__main__':
    tf.reset_default_graph()
    X_t = tf.placeholder(tf.int16, (None, 2),
                         name='x_tensor_interface')
    Z_t = tf.placeholder(tf.int16,  (None, 1),
                         name='z_tensor_interface')

    dataset = tf.data.Dataset.from_tensor_slices((X_t, Z_t))
    dataset = dataset.shuffle(buffer_size=1000)  # reshuffle_each_iteration=True as default.
    dataset = dataset.batch(2)
    dataset = dataset.flat_map(
        lambda data_x, data_z: tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensors(data_x),
                tf.data.Dataset.from_tensors(data_z),
            )
        ).repeat(3)
    )


    data_op = dataset.make_initializable_iterator()
    data_init_op = data_op.initializer
    X_batch, Z_batch = data_op.get_next()

    bias_x0 = tf.convert_to_tensor(np.array([1, 2]), dtype=tf.int16)
    bias_z0 = tf.convert_to_tensor(np.array([7]), dtype=tf.int16)

    bias_x1 = tf.convert_to_tensor(np.array([10, 11]), dtype=tf.int16)
    bias_z1 = tf.convert_to_tensor(np.array([50]), dtype=tf.int16)
    add1 = tf.nn.bias_add(X_batch, bias_x0)
    add2 = tf.nn.bias_add(Z_batch, bias_z0)
    add3 = tf.nn.bias_add(X_batch, bias_x1)
    add4 = tf.nn.bias_add(Z_batch, bias_z1)

    a = np.array([
        [100, 100],
        [200, 200],
        [300, 300],
        [400, 400],
        [500, 500],
    ])

    b = np.array([
        [600],
        [700],
        [800],
        [900],
        [600],
    ])

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        init_op.run()

        for epoch in range(10):
            print('[EPOCH]', epoch+1, '======================')
            sess.run(data_init_op, feed_dict={X_t: a, Z_t: b})
            batch_remains_ok = True
            batch_num = 0
            while batch_remains_ok and (batch_num+1 < 3):
                try:
                    for batch_num in range(3): # batch_size=2, num=3
                        print('[BATCH]', batch_num+1, '--------------')
                        res1 = sess.run(add1)
                        # res2 = sess.run(add2)
                        # res3, res4 = sess.run([add3, add4])
                        res5 = sess.run(add1)
                        # res6 = sess.run(add2)

                        print('res1', res1, '\n')
                        # print(res2)
                        # print(res3, '\n')
                        print('res5', res5, '\n')
                        # print(res6, '\n')

                except tf.errors.OutOfRangeError:
                    batch_remains_ok = False
                    continue



```


### `tf.data.TextLineDataset`

```py

file_list = glob('data/news/news_chosun*/newsdata.txt')
dataset = tf.data.Dataset.from_tensor_slices(file_list)

dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
        .skip(0)  # The first column if necessary
        .filter(
            lambda line: tf.not_equal(tf.substr(line, 0, 1), "#")
        )
    )
)

dataset = dataset.shuffle(1)
dataset = dataset.batch(2)

data_op = dataset.make_initializable_iterator()
data_init_op = data_op.initializer
next_batch = data_op.get_next()

var_init_op = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(var_init_op)
    sess.run(data_init_op)

    for _ in range(5):
        res = sess.run(next_batch)
        print(list(map(lambda x: x.decode(), res)))

```
