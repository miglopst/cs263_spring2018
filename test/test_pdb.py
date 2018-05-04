import pdb
import numpy as np
import tensorflow as tf

def get_layer(input_size, output_size, name):
    # W_val is loaded from a file using numpy.load
    W_val = np.random.normal(scale=0.1, size=(input_size, output_size)).astype(np.float32)
    W = tf.get_variable(name='W_{}'.format(name), shape=(input_size, output_size),
                        initializer=tf.constant_initializer(value=W_val, dtype=tf.float32),
                        dtype=tf.float32)
    b = tf.get_variable(name='b_{}'.format(name), shape=(output_size,),
                        initializer=tf.constant_initializer(value=0.1, dtype=tf.float32),
                        dtype=tf.float32)
    return W, b

sessions = []
for i in range(3):
    g = tf.Graph()
    with g.as_default():
        W1, b1 = get_layer(158238, 900, '1')
        W2, b2 = get_layer(900, 1000, '2')
        W3, b3 = get_layer(1000, 1, '3')

        init = tf.global_variables_initializer()
    session = tf.Session(graph=g)
    session.run(init)
    print 'Loaded {}'.format(i)
    sessions.append(session)

pdb.set_trace()
