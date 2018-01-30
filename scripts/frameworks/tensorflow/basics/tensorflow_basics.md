# Basics on Tensorflow

## Import `tensorflow`

```py
import tensorflow as tf
```

---
## The Bottom-line of `tensorflow`

> Tensorflow : `Tensor` + `Flow`.  
> It shows the key factor of `Tensorflow`.  
> This Framework have 2 steps:
> * Construct a Calculation Graph with `Tensor`s  
> * Run the session flowing the Data through the `Tensor` Structure.

### Data Types

| Data Types | Name in `Python` | Description |
| :--------- | :--------------- | :---------- |
| DT_FLOAT | `tf.float32` | 32 bits floating point. |
| DT_DOUBLE | `tf.float64` | 64 bits floating point. |
| DT_INT8 | `tf.int8` | 8 bits signed integer. |
| DT_INT16 | `tf.int16` | 16 bits signed integer. |
| DT_INT32 | `tf.int32` | 32 bits signed integer. |
| DT_INT64 | `tf.int64` | 64 bits signed integer. |
| DT_UINT8 | `tf.uint8` | 8 bits unsigned integer. |
| DT_UINT16	| `tf.uint16` | 16 bits unsigned integer. |
| DT_STRING	| `tf.string` | Variable length byte arrays. Each element of a Tensor is a byte array. |
| DT_BOOL | `tf.bool` | Boolean. |
| DT_COMPLEX64 | `tf.complex64` | Complex number made of two 32 bits floating points: <br/>real and imaginary parts. |
| DT_COMPLEX128 | `tf.complex128` | Complex number made of two 64 bits floating points: <br/>real and imaginary parts. |
| DT_QINT8 | `tf.qint8` | 8 bits signed integer used in quantized Ops. |
| DT_QINT32 | `tf.qint32` | 32 bits signed integer used in quantized Ops. |
| DT_QUINT8 | `tf.quint8` | 8 bits unsigned integer used in quantized Ops. |


  
### Data Structures
  
#### `Tensor`
__The Basic Material__ of `tensorflow`, as a container. All elements of tensorflow(inputs, parameters, outputs, etc.) is `Tensor`.  
It has `Rank`, `Shape`, and `Data Type`. You can build a model with it.

##### __Rank & Shape__
A unit of dimensionality. The number of dimensions of the `Tensor`.

| Rank | Math Entity | Example | Shape |
| :--------: | :--------------- | :---------- | :-------- |
| 0 | Scalar | s = 10 | `()` |
| 1 | Vector | v = [0, 2, 1] | `(3,)` |
| 2 | Matrix | m = [[0, 2, 1], [8, 4, 3], [9, 5, 7]]` | `(3, 3) |
| 3 | 3D-Tensor | t = [[[0, 2], [1, 8]], [[4, 3], [9, 5]], [[7, 0], [2, 1]]] | `(3, 2, 2)` |
| n | n-Tensor | n = [[[[0, 2], [1, 8]], [[4, 3], [9, 5]], [[7, 0], [2, 1]]],<br/>     [[[8, 4], [3, 9]], [[5, 7], [0, 2]], [[1, 8], [4, 3]]]] | `(2, 3, 2, 2)` |


##### Variety of Tensors

- `tf.constant` : It has only the constant values.
  ```py
  a = tf.constant([.3, .1, .2, .5, .1, .8], shape=[2, 3],
                  name='sample_constant', dtype=np.float32)
  ```

- `tf.placeholder` : It is a kind of fixed structure to flow  the data.  
                     It has only the `Shapes` and `Data Types`, not `Values` by itself.
  ```py
  a = tf.placeholder(tf.float32, shape=[None, 3],
                     name='test_placeholder)
  ```

- `tf.Variable` : It has parameters(like `weights`) and can be updated while learning. It needs to be initialized.
  ```py
  a = tf.Variable(.3, name='sample_variable')
  ```


### Computation
There is no point in having tensor itself. You should __connect tensors with functions to compute.__ If tensors are bricks, this functions are glues.

There is a formula here. Let's say all elements are scalar.
> y = 3 + 1

You can define tensors first:
```py
a = tf.constant([3.], shape=(1,), name='constant_a')
b = tf.constant([1.], shape=(1,), name='constant_b')
```

Then, you can connect the tensors with `tf.add()`:
```py
y = tf.add(a, b)
```

`y` represents a tensor, not containing the values of computation. You should run a session to get the answer.
```py
y
# <tf.Tensor 'Add_1:0' shape=(1,) dtype=float32>
```

### Session

To get the final result, You should create a session and `run()` __to flow the data through a pre-built model.__ The session will access to devices and allocate the memory to store the values and the variables.

You can create a default session to local devices:
```py
sess = tf.Session()
res = sess.run(y)
sess.close()
```

You must close the session with `sess.close()` after all computation is finished. If not, the memory in your machine keep allocated, not be released. It bothers you when you try to compute another job.

> * Note:  
> In Python, you can use `with` Statement, without concerning `sess.close()`.
> ```py
> with tf.Session() as sess:
>   res = sess.run(y)
> ```

Finally, you got the answer! the result is `numpy.ndarray`.
```py
res
# array([ 4.], dtype=float32)
```


The full script is:
```py
a = tf.constant([3.], shape=(1,), name='constant_a')
b = tf.constant([1.], shape=(1,), name='constant_b')
y = tf.add(a, b)
y
# <tf.Tensor 'Add_1:0' shape=(1,) dtype=float32>

with tf.Session() as sess:
  res = sess.run(y)
res
# array([ 4.], dtype=float32)

```

> * Note: `tf.Session.run()` vs `Tensor.eval()`
> `Tensor Objects` have a method `eval()`. This is shorthand for `tf.Session.run()` (where `tf.Session` is the current `tf.get_default_session`.


### Scope
`Scope` : The __Namespaces__ for Each `Tensor Variables` to initiate.

Global Variable Scope:
```py
init = tf.global_variables_initializer()
```

Local Variable Scope:
```py
with tf.variable_scope('foo`, reuse=True):
    v = tf.get_variable("v", [1])
```


## 

#### Using `tf.placeholder`: The way to flow the data through the tensor structure.

You can create a model with `tf.placeholder` to feed the data to the pre-built tensor model.
There is a formula for `tf.placeholder` here.
> y = x + 2


##### Fully fixed Shape (Fixed `n` samples)

You can define tensors first. The shape is defined  `(n_samples, ...)`.
The shape is `(1, )`, which means __only one sample can be evaluated at a time.__
```py
data = np.array([.4])

x = tf.placeholder(tf.float32, shape=(1,), name='x')
b = tf.constant([2.], shape=(1,), name='constant_b')
y = tf.add(x, b)

with tf.Session() as sess:
  res = sess.run(y, feed_dict={x: data})
res
# array([6.], dtype=float32
```

#####  Dynamic Shape (Dynamic `n` samples)

You can define tensors with `None` to accept dynamic input numbers. The shape is defined as `(None, ...)` at that case.
```py
data = np.array([.4, 1.5, -.3])

x = tf.placeholder(tf.float32, shape=(None,), name='x')
b = tf.constant([2.], shape=(1,), name='constant_b')
y = tf.add(x, b)

with tf.Session() as sess:
  res = sess.run(y, feed_dict={x: data})
res
# array([2.4, 3.5, 1.7], dtype=float32)
```





#### `tf.device()`


#### `tf.Graph()`


---

    - [Constants]()
    - [Sequences]()
    - [Randoms]()
    
  
  - [Data Structures]()
    - [Tensor Transformations]()
      - [Casting]()
      - [Shaping]()
      - [Slicing & Joining]()
      - [Casting]()

* [Control Flow]()
  - [Operators]()
    - [Logical Operators]()
    - [Comparison Operators]()
    - [Debugging]()

* [Operations(Math)]()
  - [Arithmetics]()
  - [Basic Functions]()
  - [Matrix Functions]()
  - [Tensor Functions]()
  - [Complex Number Functions]()
  - [Functions for Reduction]()
  - [Scan(total, cumulative)]()
  - [Segmentation]()
  - [Sequence Comparison and Indexing]()

Layers

```py
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)
```

Initializing Layers

Global

```py
init = tf.global_variables_initializer()
sess.run(init)
```

