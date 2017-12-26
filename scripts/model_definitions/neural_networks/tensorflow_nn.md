## Tensorflow NN

```py
import tensorflow as tf

class MyNN:
    
    def __init__(self, n_in, n_hiddens, n_out):
        
        self.n_in = n_in
        self.n_hiddens = n.hiddens
        self.n_out = n_out
        
        self.weights = []
        self.biases = []
        
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=.01)
        return tf.Variabel(initial)
    
    def bias_variable(self, shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)
        
    # Define a Model
    def inference(self, x, keep_prob):
        y = x
        return y
        
    def loss(self, y, t):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1])
        return cross_entropy
    
    def training(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(.01)
        train_step = optimizer.minimize(loss)
        return train_step
        
    def accuracy(self, y, t):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)
        return accuracy
        
    def fit(self, X_train, Y_train):
        
        
    def evaluate(self, X_test, Y_test):
    
        
```

```py
import tensorflow as tf

class MyNN:
    
    def __init__(self, n_in, n_hiddens, n_out):
        
        self.n_in = n_in
        self.n_hiddens = n.hiddens
        self.n_out = n_out
        
        self.weights = []
        self.biases = []
        
        self._x = None
        self._t = None
        self._keep_prob = None
        self._sess = None
        self._history = {'accuracy': [], 'loss': []}
        
        
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=.01)
        return tf.Variabel(initial)
    
    
    def bias_variable(self, shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)
        
        
    # Define a Model
    def inference(self, x, keep_prob):
        
        # INPUT to HIDDEN, HIDDEN to HIDDEN
        for i, n_hidden in enumerate(self, n_hiddens):
            if i == 0:
                input = x
                input_dim = self.n_in
            else:
                input = output
                input_dim = self.n_hiddens[i-1]
            
            self.weights.append(self.weight_variable([input_dim, n_hidden])
            self.biases.append(self.bias_variable([n_hidden]))
            
            h = tf.nn.reul(tf.matmul(input, self.weights[-1]) + self.biases[-1])
            output = tf.nn.dropout(h, keep_prob)
        
        # HIDDEN to OUTPUT
        self.weights.append(self.weight_variable([self.n_hiddens[-1], self.n_out])
        self.biases.append(self.bias_variable([self.n_out]))
        
        y = tf.nn.sigmoid(tf.matmul(output, self.weights[-1]) + self.biases[-1])
        
        return y
        
    def loss(self, y, t):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1])
        return cross_entropy
    
    def training(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(.01)
        train_step = optimizer.minimize(loss)
        return train_step
        
    def accuracy(self, y, t):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)
        return accuracy
        
    def fit(self, X_train, Y_train, epochs=100, batch_size=16, p_keep=.7, verbose=True):
        x = tf.placeholder(tf.float32, shape=[None, self.n_in])
        t = tf.placeholder(tf.float32, shape=[None, self.n_out])
        keep_prob = tf.placeholder(tf.float32)
        
        self._x = x
        self._t = t
        self._keep_prob = keep_prob
        
        y = self.inference(x, keep_prob)
        loss = self.loss(y, t)
        train_step = self.training(loss)
        accuracy = self.accuracy(y, t)
        
        init = tf.global_variables_initialiser()
        selff = tf.Session()
        sess.run(init)
        
        
        self._sess = sess
        
        N_train = len(X_train)
        n_bathes = N_train // batch_size
        
        for epoch in range(epochs):
            X_, Y_ = shuffle(X_train, Y_train)

            for i in range(n_batches):
               start = i * batch_size
               end = start + batch_size

               sess.run(train_step, feed_dict={x: X_[start:end], t: Y_[start:end], keep_prob=p_keep})

            loss_ = loss.eval(session=sess, feed_dict={x: X_[start:end], t: Y_[start:end], keep_prob=1.})
            accuracy_ = accuracy.eval(session=sess, feed_dict={x: X_[start:end], t: Y_[start:end], keep_prob=1.})

            self._history['loss'].append(loss_)
            self._history['accuracy'].append(accuracy_)


            if verbose:
                print('epoch:', epoch, '\tloss:', loss_, '\taccuracy:', accuracy_)
        
        return self._history
        
        
    def evaluate(self, X_test, Y_test):
        res = self.accuracy.eval(session=self._sess, feed_dict={self._x = X_test, self._t: Y_test, self._keep_prob: 1.})
        return res
        
```
