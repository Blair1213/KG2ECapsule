# -*- coding: utf-8 -*-
# @Time    : 2021/9/16 下午10:43
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : capsule.py
# @Software : PyCharm

from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf

def squash(x,axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


class Caps(Layer):
    def __init__(self,capsule_dim,capsule_num, activation = 'default',initializer='glorot_normal',routings = 3 ,kernel_size = (9,1),regularizer=None, share_weights = True,
                 **kwargs):
        super(Caps,self).__init__(**kwargs)
        self.capsule_num = capsule_num
        self.capsule_dim = capsule_dim
        self.initializer = initializer
        self.regularizer = regularizer
        self.share_weights = share_weights
        self.routings = routings
        self.kernel_size = kernel_size
        if activation == 'default':
            self.activation = squash

    def build(self, input_shape):

        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]
        super(Caps, self).build(input_shape)
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(self.input_num_capsule,self.capsule_num,
                                            self.input_dim_vector,self.capsule_dim), ##[1,hop,4*32] capsule_dim = 32
                                     initializer=self.initializer)
            self.b = self.add_weight(shape=[1, self.input_num_capsule, self.capsule_num, 1, 1],
                                     initializer=self.initializer,
                                     name='b',
                                     trainable=False)
            self.built = True



    def call(self,inputs, training=None):
        ##[batch,hop,dim]->[batch,hop-1,dim]

        print(inputs.shape)

        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)
        print(inputs_expand)
        # Replicate num_capsule dimension to prepare being multiplied by W
        # Now shape = [None, input_num_capsule, num_capsule, 1, input_dim_vector]
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.capsule_num, 1, 1])
        print(inputs_tiled)

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = tf.scan(lambda ac, x: tf.linalg.matmul(x, self.W),
                             elems=inputs_tiled,
                             initializer=K.zeros([self.input_num_capsule, self.capsule_num, 1, self.capsule_dim]))
        print("input_hat")
        # Routing algorithm
        assert self.routings > 0, 'The num_routing should be > 0.'
        for i in range(self.routings):
            c = tf.nn.softmax(self.b, dim=2)  # dim=2 is the num_capsule dimension
            # outputs.shape=[None, 1, num_capsule, 1, dim_vector]
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            # last iteration needs not compute b which will not be passed to the graph any more anyway.
            if i != self.routings - 1:
                self.b = self.b + K.sum(inputs_hat * outputs, -1, keepdims=True)

        return K.reshape(outputs, [-1, self.capsule_num, self.capsule_dim])



    def compute_output_shape(self, input_shape):
        return (None, self.capsule_num, self.capsule_dim)

