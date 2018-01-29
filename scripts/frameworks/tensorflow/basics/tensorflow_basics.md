# Basics on Tensorflow

## Import `tensorflow`

```py
import tensorflow as tf
```

## The Bottom-line of `tensorflow`

Tensorflow : `Tensor` + `Flow`.  
It shows the key factor of `Tensorflow`.  
This Framework have 2 steps:
* Construct a Calculation Graph with `Tensor`s  
* Run the session flowing the Data through the `Tensor` Structure.


* `Tensor` : The __Containers__ for Data.

```py
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)
```

Result:
```sh
Tensor("Const:0", shape=(), dtype=float32)
Tensor("Const_1:0", shape=(), dtype=float32)
Tensor("add:0", shape=(), dtype=float32)
```


* `Scope` : The __Local Namespaces__ for Each `Tensor Variables`.

Global Variable Scope:
```py
init = tf.global_variables_initializer()
```

Local Variable Scope:
```py
with tf.variable_scope('foo`, reuse=True):
    v = tf.get_variable("v", [1])
```

* `Session`: Running the processor to __Flow__ the Data to the `Tensor Graph`, which is built by `Tensors` and `Scopes`.
  - `tf.Session()`

```py
sess = tf.Session()
print(sess.run(*args))
sess.close()
```

```py
with tf.Session() as sess:
  print(sess.run(*args))
```

  - `tf.device()`


  - `tf.Graph()`


## Variables

  - [Tensor]()
    - `tf.Variable`
    ```py
    a = tf.Variable(.3, name='sample_variable')
    ```
    
    - `tf.constant`
    ```py
    a = tf.constant([.3, .1, .2, .5, .1, .8], shape=[2, 3],
                    name='sample_constant', dtype=np.float32)
    ```
    
    - `tf.placeholder`
    ```py
    a = tf.placeholder(tf.float32, shape=[None, 3],
                       name='test_placeholder)
    ```
  - [Scope]()
  
  - [Session]()
    - [`tf.Session.run()` vs `Tensor.eval()`]()

* [Basic Objects]()
  - [Data Types]()

| Data Types | Name in `Python` | Description |
| :--------- | :--------------- | :---------- |
| DT_FLOAT | tf.float32 | 32 bits floating point. |
| DT_DOUBLE | tf.float64 | 64 bits floating point. |
| DT_INT8 | tf.int8 | 8 bits signed integer. |
| DT_INT16 | tf.int16 | 16 bits signed integer. |
| DT_INT32 | tf.int32 | 32 bits signed integer. |
| DT_INT64 | tf.int64 | 64 bits signed integer. |
| DT_UINT8 | tf.uint8 | 8 bits unsigned integer. |
| DT_UINT16	| tf.uint16 | 16 bits unsigned integer. |
| DT_STRING	| tf.string | Variable length byte arrays. Each element of a Tensor is a byte array. |
| DT_BOOL | tf.bool | Boolean. |
| DT_COMPLEX64 | tf.complex64 | Complex number made of two 32 bits floating points: real and imaginary parts. |
| DT_COMPLEX128 | tf.complex128 | Complex number made of two 64 bits floating points: real and imaginary parts. |
| DT_QINT8 | tf.qint8 | 8 bits signed integer used in quantized Ops. |
| DT_QINT32 | tf.qint32 | 32 bits signed integer used in quantized Ops. |
| DT_QUINT8 | tf.quint8 | 8 bits unsigned integer used in quantized Ops. |


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

