import tensorflow as tf

def Conv_layer(names, input, w_shape, b_shape, strid):
    with tf.variable_scope(names) as scope:
        weights = tf.Variable(tf.truncated_normal(shape = w_shape, stddev = 1.0, dtype = tf.float32),
                              name = 'weights_{}'.format(names), dtype = tf.float32)

        biases = tf.Variable(tf.constant(value = 0.1, dtype = tf.float32, shape = b_shape),
                             name='biases_{}'.format(names), dtype = tf.float32)

        conv = tf.nn.conv2d(input, weights, strides = [1, strid[0], strid[1], 1], padding = 'SAME')
        # print(strid)
        conv = tf.nn.bias_add(conv, biases)
        conv_out = tf.nn.relu(conv, name = 'relu_{}'.format(names))

        # print("---------names:{}".format(conv_out))
        return conv_out

def Max_pool_lrn(names, input, ksize, is_lrn):
    with tf.variable_scope(names) as scope:
        Max_pool_out = tf.nn.max_pool(input, ksize = ksize, strides = [1, 2, 2, 1], padding = 'SAME', name = 'max_pool_{}'.format(names))
        if is_lrn:
            Max_pool_out = tf.nn.lrn(Max_pool_out, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name = 'lrn_{}'.format(names))
            # print("use lrn operation")
    return Max_pool_out



def Dropout_layer(names, input, drop_rate):
    with tf.variable_scope(names) as scope:
        #drop_out =local3
        drop_out = tf.nn.dropout(input, drop_rate)
    return drop_out

def local_layer(names, input, w_shape, b_shape):
    with tf.variable_scope(names) as scope:
        weights = tf.Variable(tf.truncated_normal(shape = w_shape, stddev=0.005, dtype = tf.float32),
                              name = 'weights_{}'.format(names), dtype = tf.float32)

        biases = tf.Variable(tf.constant(value = 0.1, dtype = tf.float32, shape = b_shape),
                             name='biases_{}'.format(names), dtype=tf.float32)

        local = tf.nn.relu(tf.matmul(input, weights) + biases, name ='local_{}'.format(names))
    return local

def inference(images, batch_size, n_classes):#,drop_rate

    conv1 = Conv_layer(names = 'conv_block1', input = images , w_shape = [3, 3, 3, 64], b_shape = [64], strid = [1, 1])
    conv2 = Conv_layer(names = 'conv_block2', input = conv1 , w_shape = [3, 3, 64, 64], b_shape = [64], strid = [1, 1])
    pool_1 = Max_pool_lrn(names = 'pooling1', input = conv2 , ksize = [1, 2, 2, 1], is_lrn = True)
    print('pool_1',pool_1.shape) #(?, 112, 112, 64)
    conv3 = Conv_layer(names = 'conv_block3', input = pool_1 , w_shape = [3, 3, 64, 128], b_shape = [128], strid = [1, 1])
    conv4 = Conv_layer(names = 'conv_block4', input = conv3 , w_shape = [3, 3, 128, 128], b_shape = [128], strid = [1, 1])
    pool_2 = Max_pool_lrn(names = 'pooling2', input = conv4 , ksize = [1, 2, 2, 1], is_lrn = False)
    print('pool_2',pool_2.shape) #(?, 56, 56, 128)
    conv5 = Conv_layer(names = 'conv_block5', input = pool_2 , w_shape = [3, 3, 128, 256], b_shape = [256], strid = [1, 1])
    conv6 = Conv_layer(names = 'conv_block6', input = conv5 , w_shape = [3, 3, 256, 256], b_shape = [256], strid = [1, 1])
    conv7 = Conv_layer(names = 'conv_block7', input = conv6 , w_shape = [3, 3, 256, 256], b_shape = [256], strid = [1, 1])
    pool_3 = Max_pool_lrn(names = 'pooling3', input = conv7 , ksize = [1, 2, 2, 1], is_lrn = False)
    print('pool_3',pool_3.shape) #(?, 28, 28, 256)
    conv8 = Conv_layer(names = 'conv_block8', input = pool_3 , w_shape = [3, 3, 256, 512], b_shape = [512], strid = [1, 1])
    conv9 = Conv_layer(names = 'conv_block9', input = conv8 , w_shape = [3, 3, 512, 512], b_shape = [512], strid = [1, 1])
    conv10 = Conv_layer(names = 'conv_block10', input = conv9 , w_shape = [3, 3, 512, 512], b_shape = [512], strid = [1, 1])
    pool_4 = Max_pool_lrn(names = 'pooling4', input = conv10 , ksize = [1, 2, 2, 1], is_lrn = False)
    print('pool_4',pool_4.shape) #(?, 14, 14, 512)
    conv11 = Conv_layer(names = 'conv_block11', input = pool_4 , w_shape = [3, 3, 512, 512], b_shape = [512], strid = [1, 1])
    conv12 = Conv_layer(names = 'conv_block12', input = conv11 , w_shape = [3, 3, 512, 512], b_shape = [512], strid = [1, 1])
    conv13 = Conv_layer(names = 'conv_block13', input = conv12 , w_shape = [3, 3, 512, 64], b_shape = [64], strid = [1, 1])
    pool_5 = Max_pool_lrn(names = 'pooling5', input = conv13 , ksize = [1, 2, 2, 1], is_lrn = False)
    print('pool_5',pool_5.shape) #(?, 7, 7, 64)
    
    #reshape = tf.reshape(pool_5, shape=[batch_size, -1]) #(128, ?)
    shape = pool_5.get_shape().as_list()
    print('shape',shape)# 
    dim = 1
    for i in range(1,len(shape)):
      dim*=shape[i]
    pool_5reshape = tf.reshape(pool_5, [-1, dim])
    dim = pool_5reshape.get_shape()[1].value
    print('dim',dim) #None
    local_1 = local_layer(names = 'local1_scope', input = pool_5reshape , w_shape = [dim, 1024], b_shape = [1024])
    print('local_1',local_1.shape)
    local_2 = local_layer(names = 'local2_scope', input = local_1 , w_shape = [1024, 1024], b_shape = [1024])
    print('local_2',local_2.shape)

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[1024, n_classes], stddev=0.005, dtype=tf.float32),
                              name='softmax_linear', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[n_classes]),
                             name='biases', dtype=tf.float32)

        softmax_linear = tf.add(tf.matmul(local_2, weights), biases, name='softmax_linear')
        # print("---------softmax_linear:{}".format(softmax_linear))

    return softmax_linear


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        # print("\ncross_entropy:{},cross_entropy.shape:{}".format(cross_entropy,cross_entropy.shape))
        # print("---------cross_entropy:{}".format(cross_entropy))
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
