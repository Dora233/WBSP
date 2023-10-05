# -*- coding: utf-8 -*-

import tensorflow as tf

TRAINING = tf.Variable(initial_value=True, dtype=tf.bool, trainable=False)

def identity_block(X_input, kernel_size, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("id_block_stage"+str(stage)):
        filter1, filter2, filter3 = filters
        X_shortcut = X_input

        # First component of main path
        x = tf.layers.conv2d(X_input, filter1,
                 kernel_size=(1, 1), strides=(1, 1),name=conv_name_base+'2a')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'2a', training=TRAINING)
        x = tf.nn.relu(x)

        # Second component of main path
        x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size),
                                 padding='same', name=conv_name_base+'2b')
        # batch_norm2 = tf.layers.batch_normalization(conv2, axis=3, name=bn_name_base+'2b', training=TRAINING)
        x = tf.nn.relu(x)

        # Third component of main path
        x = tf.layers.conv2d(x, filter3, kernel_size=(1, 1),name=conv_name_base+'2c')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=TRAINING)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(x, X_shortcut)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result


def convolutional_block(X_input, kernel_size, filters, stage, block, stride = 2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    stride -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("conv_block_stage" + str(stage)):

        # Retrieve Filters
        filter1, filter2, filter3 = filters

        # Save the input value
        X_shortcut = X_input

        # First component of main path
        x = tf.layers.conv2d(X_input, filter1,
                                 kernel_size=(1, 1),
                                 strides=(stride, stride),
                                 name=conv_name_base+'2a')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'2a', training=TRAINING)
        x = tf.nn.relu(x)

        # Second component of main path
        x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size), name=conv_name_base + '2b',padding='same')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2b', training=TRAINING)
        x = tf.nn.relu(x)

        # Third component of main path
        x = tf.layers.conv2d(x, filter3, (1, 1), name=conv_name_base + '2c')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=TRAINING)

        # SHORTCUT PATH
        X_shortcut = tf.layers.conv2d(X_shortcut, filter3, (1,1),
                                      strides=(stride, stride), name=conv_name_base + '1')
        X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, name=bn_name_base + '1', training=TRAINING)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(X_shortcut, x)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result


def ResNet50_reference(X, classes= 1000):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:

    Returns:
    """

    #x = tf.pad(X, tf.constant([[0, 0],[3, 3,], [3, 3], [0, 0]]), "CONSTANT")

    #assert(x.shape == (x.shape[0], 70, 70, 3))
    x = X

    # stage 1
    x = tf.layers.conv2d(x, filters=32, kernel_size=(7, 7), strides=(2, 2), name='conv1')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_conv1')
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=(3, 3),strides=(2, 2))
    print('stage 1 shape',x.shape) #(?, 54, 54, 32)

    # stage 2
    x = convolutional_block(x, kernel_size=3, filters=[32, 32, 128], stage=2, block='a', stride=1)
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='b')
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='c')
    print('stage 2 shape',x.shape) # (?, 54, 54, 128)
    # stage 3
    x = convolutional_block(x, kernel_size=3, filters=[64,64,256],
                                            stage=3, block='a', stride=2)
    x = identity_block(x, 3, [64,64,256], stage=3, block='b')
    x = identity_block(x, 3, [64,64,256], stage=3, block='c')
    x = identity_block(x, 3, [64,64,256], stage=3, block='d')
    print('stage 3 shape',x.shape) #(?, 27, 27, 256)
    
    # stage 4
    x = convolutional_block(x, kernel_size=3, filters=[128, 128, 512], stage=4, block='a', stride=2)
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='d')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='e')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='f')
    print('stage 4 shape',x.shape) #(?, 14, 14, 512)
    
    # stage 5
    x = convolutional_block(x,kernel_size=3,filters=[256, 256, 64], stage=5, block='a', stride=2)
    x = identity_block(x, 3, [256, 256, 64], stage=5, block='b')
    x = identity_block(x, 3, [256, 256, 64], stage=5, block='c')
    print('stage 5 shape',x.shape) #(?, 7, 7, 64)
    
    x = tf.layers.average_pooling2d(x, pool_size=(2, 2), strides=(1,1))
    print('avgpool shape',x.shape) #(?, 6, 6, 64)
    
    flatten = tf.layers.flatten(x, name='flatten')
    print('flatten shape',x.shape) #(?, 6, 6, 64)

    dense1 = tf.layers.dense(flatten, units=1024, activation=tf.nn.relu)
    logits = tf.layers.dense(dense1, units=classes, activation=tf.nn.softmax)
    return logits
