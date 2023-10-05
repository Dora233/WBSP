#CUDA_VISIBLE_DEVICES= nohup python -u run.py --class_num=1000  --model=alexnet  --job_name=ps --worker_num=2 --base_dir=osp/ --port_base=2448 >ps.log 2>&1 & 
#CUDA_VISIBLE_DEVICES=0 nohup python -u run.py --class_num=1000  --model=alexnet  --job_name=worker --worker_index=0 --base_dir=osp/ --port_base=2448 >wk0.log 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --class_num=1000  --model=alexnet  --job_name=worker --worker_index=1 --base_dir=osp/ --port_base=2448 >wk1.log 2>&1 &

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2,3" #,4,5,6,7
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息
# TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息
# TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息
# TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息
import time
import numpy as np
import tensorflow as tf
#import load_cifar10
from create_tf_record import *
from inceptionv3_zzc import *
import tensorflow.contrib.slim as slim
#SAVE_VARIABLES = 'save_variables'

import sys
import argparse


config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.33  
config.gpu_options.allow_growth = True   
sess = tf.Session(config = config)

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
#flags.DEFINE_string('data_dir', '/data/backup_data/lidaox/gyq/dataset/tiny-imagenet-200/tiny-imagenet-200/record', """Path to the CIFAR-10 data directory.""")
flags.DEFINE_string('data_dir', '/home/zjlab/yangduo/cifar10_data', """Path to the CIFAR-10 data directory.""")
flags.DEFINE_integer('BUFF_SIZE', 1024, """Number of images to process in a batch.""")                      
# both worker and ps
flags.DEFINE_string('base_dir', 'pcc/', 'The path where log info will be stored')
flags.DEFINE_string('model', 'alexnet', 'The inference')
#/lidaox/yangduo/TensorFlow/STrain-main/ours_cifar10/base/    
flags.DEFINE_string('host', 'localhost', 'IP address of parameter server')
flags.DEFINE_integer('port_base', 2222, 'Start port for listening to workers')
flags.DEFINE_string('job_name', '', 'Either ps or worker')
flags.DEFINE_integer('class_num', 1000, 'Training batch size ')
flags.DEFINE_float('check_period', 5.0, 'Length of time between two checkpoints')
# wk
flags.DEFINE_integer('worker_index', 0, 'Index of worker')
flags.DEFINE_float('base_time_step', 1.0, 'Initial length of time between two commits for each worker')
flags.DEFINE_integer('max_steps', 1000000, 'Max steps each worker is allowed to run')
flags.DEFINE_float('sleep_time', 1.0, 'Specify the sleep_time')
# ps
flags.DEFINE_integer('worker_num', 8, 'Total number of workers')
flags.DEFINE_float('band_width_limit', None, 'Specify the constrait of bandwidth in the form of x M/s')
flags.DEFINE_float('training_end', 0.1, 'When loss is smaller than this, end training')
flags.DEFINE_float('epsilon', 0.3, 'Epsilon')
FLAGS = flags.FLAGS

if(FLAGS.class_num==200):
    train_record_file='/data/backup_data/imagenet2012_tfrecords/lidaox/gyq/dataset/tiny-imagenet-200/tiny-imagenet-200/record/train299.tfrecords'
    val_record_file='/data/backup_data/imagenet2012_tfrecords/lidaox/gyq/dataset/tiny-imagenet-200/tiny-imagenet-200/record/val299.tfrecords'
if(FLAGS.class_num==1000):
    train_record_file='/data/backup_data/imagenet2012_1000_tfrecords/record/train299.tfrecords'
    val_record_file='/data/backup_data/imagenet2012_1000_tfrecords/record/val299.tfrecords'

# NUM_CLASSES = 10

labels_nums = FLAGS.class_num

resize_height = 299
resize_width = 299
depths = 3

#data_shape = [FLAGS.batch_size, resize_height, resize_width, depths]

if (FLAGS.base_dir =='asp/'):
    import wk_ssp as st_wk
    import ps_ssp as st_ps
    flags.DEFINE_integer('s', 1000, 'threshold')
if (FLAGS.base_dir =='ssp/'):
    import wk_ssp as st_wk
    import ps_ssp as st_ps
    flags.DEFINE_integer('s', 40, 'threshold')
if (FLAGS.base_dir =='bsp/'):
    import wk_ssp as st_wk
    import ps_ssp as st_ps
    flags.DEFINE_integer('s', 1, 'threshold')
if (FLAGS.base_dir =='adsp/'):
    import wk_adsp as st_wk
    import ps_adsp as st_ps
    flags.DEFINE_integer('s', 0, 'threshold')
if (FLAGS.base_dir =='wbsp/'):
    import wk_wbsp as st_wk
    import ps_wbsp as st_ps
    flags.DEFINE_integer('s', 0, 'threshold')
if (FLAGS.base_dir =='pcc/'):
    import wk_pcc_v2 as st_wk
    import ps_wbsp as st_ps
    flags.DEFINE_integer('s', 0, 'threshold')
if (FLAGS.base_dir =='wbsplocal/'):
    import wk_wbsplocal as st_wk
    import ps_wbsplocal as st_ps
    flags.DEFINE_integer('s', 0, 'threshold')
if (FLAGS.base_dir =='pcclocal/'):
    import wk_pcclocal as st_wk
    import ps_wbsplocal as st_ps
    flags.DEFINE_integer('s', 0, 'threshold')

def image_alexnet(images):
    #conv1
    with tf.name_scope('conv1') as scope:
        kernel=tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,
                                            stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME') # stride=4
        print('conv shape',conv.shape) #(?, 56, 56, 64)
        biases=tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),
                                        trainable=True,name='bias')
        bias=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(bias,name=scope) 
    tf.summary.histogram('Convolution_layers/conv1',conv1)
    tf.summary.scalar('Convolution_layers/conver1',tf.nn.zero_fraction(conv1))
    #lrn1
    with tf.name_scope('lrn1') as scope:
        lrn1=tf.nn.local_response_normalization(conv1,alpha=1e-4,beta=0.75,
                                                depth_radius=2,bias=2.0)
    #pool1
    pool1=tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1], 
                    padding='VALID',name='pool1')
    print('pool1 shape',pool1.shape) #(?, 27, 27, 64)

    #conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 64], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                                 trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        print('conv2 shape',conv2.shape)#(?, 27, 27, 64)
    tf.summary.histogram('Convolution_layers/conv2',conv2)
    tf.summary.scalar('Convolution_layers/conver2',tf.nn.zero_fraction(conv2))
    #lrn2
    with tf.name_scope('lrn2') as scope:
        lrn2 = tf.nn.local_response_normalization(conv2,alpha=1e-4,beta=0.75,
                                                depth_radius=2, bias=2.0)
    # pool2
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],
                            padding='VALID',name='pool2')
    print('pool2 shape',pool2.shape) # 13 13 64
    #conv3
    with tf.name_scope('conv3') as scope:
        kernel =tf.Variable(tf.truncated_normal([3,3,64,128],dtype=tf.float32,
                                                stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[128],dtype=tf.float32),
                                        trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv3=tf.nn.relu(bias,name=scope)
        print('conv3 shape',conv3.shape) 
    tf.summary.histogram('Convolution_layers/conv3',conv3)
    tf.summary.scalar('Convolution_layers/conver3',tf.nn.zero_fraction(conv3))
    #conv4
    with tf.name_scope('conv4') as scope:
        kernel =tf.Variable(tf.truncated_normal([3,3,128,128],dtype=tf.float32,
                                                stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[128],dtype=tf.float32),
                                        trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv4=tf.nn.relu(bias,name=scope)
        print('conv4 shape',conv4.shape) 
    tf.summary.histogram('Convolution_layers/conv4',conv4)
    tf.summary.scalar('Convolution_layers/conver4',tf.nn.zero_fraction(conv4))
    #conv5
    with tf.name_scope('conv5') as scope:
        kernel =tf.Variable(tf.truncated_normal([3,3,128,64],dtype=tf.float32,
                                                stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),
                                        trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv5=tf.nn.relu(bias,name=scope)
        print('conv5 shape',conv5.shape)  #13 13 64
    tf.summary.histogram('Convolution_layers/conv5',conv5)
    tf.summary.scalar('Convolution_layers/conver5',tf.nn.zero_fraction(conv5))

    #pool5
    pool5=tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],
                            padding='VALID',name='pool5')
    print('pool5 shape',pool5.shape) #6 6 64

    #fully_connected1
    with tf.name_scope('fully_connected1') as scope:
        shape = pool5.get_shape().as_list()
        print('shape',shape)# [None, 6, 6, 64]
        dim = 1
        for i in range(1,len(shape)):
          dim*=shape[i]
        pool5_2 = tf.reshape(pool5, [-1, dim])
        dim=pool5_2.get_shape()[1].value
        print('pool5_2 shape',pool5_2.shape) # (?, 2304)
        print('dim',dim)#2304
        weights =tf.Variable(tf.truncated_normal([dim,512],dtype=tf.float32,
                                                stddev=1e-1),name='weights')
        biases=tf.Variable(tf.constant(0.0,shape=[512],dtype=tf.float32),
                                        trainable=True,name='biases')
        local3=tf.nn.relu(tf.matmul(pool5_2,weights)+biases,name=scope)
    tf.summary.histogram('Fully connected layers/fc1',local3)
    tf.summary.scalar('Fully connected layers/fc1',tf.nn.zero_fraction(local3))

    #fully_connected2
    with tf.name_scope('fully_connected') as scope:
        weights =tf.Variable(tf.truncated_normal([512,384],dtype=tf.float32,
                                                stddev=1e-1),name='weights')
        biases=tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),
                                        trainable=True,name='biases')
        local4=tf.nn.relu(tf.matmul(local3,weights)+biases,name=scope)
    tf.summary.histogram('Fully connected layers/fc2',local4)
    tf.summary.scalar('Fully connected layers/fc4',tf.nn.zero_fraction(local4))

    #output
    with tf.name_scope('output') as scope:
        weights =tf.Variable(tf.truncated_normal([384,FLAGS.class_num],dtype=tf.float32,
                                                stddev=1e-1),name='weights')
        biases=tf.Variable(tf.constant(0.0,shape=[FLAGS.class_num],dtype=tf.float32),
                                        trainable=True,name='biases')
        softmax_linear=tf.add(tf.matmul(local4,weights),biases,name=scope)
    tf.summary.histogram('Fully connected layers/output',softmax_linear)

    #global_step=tf.Variable(initial_value=0,name='global_step',trainable=False)
    #y_pred_cls=tf.argmax(softmax_linear,axis=1)                
    
    return softmax_linear

def build_model(device):
    global sess, merged, global_step, images, labels, is_training_ph, train_X, train_Y,train_X2, train_Y2 
    global lr, train_op, eval_ops, loss, loss_ps, top_1_op,top_5_op,top_1_op_ps, top_5_op_ps
    with tf.device(device):
        images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths])
        #print('images',images.shape) images (?, 224, 224, 3) labels (?, 200)
        labels = tf.placeholder(dtype=tf.float32, shape=[None, labels_nums])
        is_training_ph= tf.placeholder(dtype=tf.bool, name='is_training')
        
        train_images, train_labels = read_records(train_record_file, resize_height, resize_width, type='normalization')
        train_X, train_Y = get_batch_images(train_images, train_labels,
                                              batch_size=FLAGS.batch_size, labels_nums=labels_nums,
                                              one_hot=True, shuffle=True)
        val_images, val_labels = read_records(val_record_file, resize_height, resize_width, type='normalization')
        train_X2, train_Y2 = get_batch_images(val_images, val_labels,
                                              batch_size=FLAGS.batch_size, labels_nums=labels_nums,
                                              one_hot=True, shuffle=False)
        print('train_X',train_X.shape) # (128, 224, 224, 3)
        print('train_Y',train_Y.shape) # (128, 1000)
        # global_step = tf.get_variable('global_step', [], dtype= tf.int32, initializer= tf.constant_initializer(0), trainable= False, 
            # collections=[tf.GraphKeys.GLOBAL_VARIABLES, SAVE_VARIABLES])
        global_step = tf.train.get_or_create_global_step()
        tf.summary.scalar('global_step', global_step)
        
        # dev_CPLEX_2GPU = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
                      # 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1,
                      # 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0,
                      # 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1]
        dev_CPLEX_2GPU = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #out = image_alexnet(images)
        out, end_points = inception_v3(inputs = images,  dev=dev_CPLEX_2GPU, num_classes = labels_nums, is_training=is_training_ph)
        
        tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=out) #loss=1.6
        loss = tf.losses.get_total_loss(add_regularization_losses=True)
        
        # out = slim.softmax(out)
        # # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            # # logits=out, labels=tf.argmax(labels,1), name='cross_entropy_per_example')
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            # logits=out, labels=labels, name='cross_entropy_per_example')
        
        # cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        # regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # loss = tf.add_n([cross_entropy_mean] + regu_losses)
        # #ema = tf.train.ExponentialMovingAverage(0.9, global_step)
        # #tf.add_to_collection('resnet_update_ops', ema.apply([loss]))
        
        # # #--------------------------------------------
        # # predictions = tf.nn.softmax(out)
        # # top_1_op = top_k_error(predictions, labels, 1)
        # # top_5_op = top_k_error(predictions, labels, 5)
    
        # # tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=out)
        # # loss = tf.losses.get_total_loss(add_regularization_losses=True)
        # #predictions = tf.nn.softmax(out)
        # predictions = out
        # top_1_op = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions, tf.argmax(labels,1), 1),tf.float32))
        # top_5_op = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions, tf.argmax(labels,1), 5),tf.float32))
        
        
        decay_steps = int(1280*FLAGS.class_num/FLAGS.batch_size)
        boundaries = [round(5 * decay_steps), round(10 * decay_steps)]
        learing_rates = [0.01, 0.01, 0.01]
        lr = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learing_rates)
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        #opt = tf.train.AdamOptimizer(learning_rate=lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = slim.learning.create_train_op(total_loss=loss, optimizer=opt)
        top_1_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(labels, 1)), tf.float32))
        top_5_op = tf.reduce_mean(tf.cast(tf.nn.in_top_k(out, tf.argmax(labels,1), 5),tf.float32))
        # grads = opt.compute_gradients(loss)
        # #for i, (g, v) in enumerate(grads):
        # #    if g is not None:
        # #        grads[i] = (tf.clip_by_norm(g, 10), v)
        # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        # #batchnorm_updates = tf.get_collection('resnet_update_ops')
        # #batchnorm_updates_op = tf.group(*batchnorm_updates)
        # #train_op = tf.group(apply_gradient_op, batchnorm_updates_op)
        # train_op = apply_gradient_op
        
        saver = tf.train.Saver(tf.all_variables())
        #--------------------------------------------
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        merged = tf.summary.merge_all()	
        sess.run(init)
        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    
def process_worker(device):
    global sess, merged, global_step, images, labels, is_training_ph, train_X, train_Y,train_X2, train_Y2 
    global lr, train_op, eval_ops, loss, loss_ps, top_1_op,top_5_op,top_1_op_ps, top_5_op_ps
    strain_wk = st_wk.STRAIN_WK(
            _worker_index = FLAGS.worker_index, 
            _check_period = FLAGS.check_period, 
            _init_base_time_step = FLAGS.base_time_step, 
            _max_steps = FLAGS.max_steps, 
            _batch_size = FLAGS.batch_size, 
            _class_num = FLAGS.class_num,
            _base_dir = FLAGS.base_dir,
            _host = FLAGS.host,
            _port_base = FLAGS.port_base,
            _buffsize = FLAGS.BUFF_SIZE)
    build_model(device)
    #print('---------------------build_model is OK-------------------')
    strain_wk.register(
        _session = sess,
        _data_dir = FLAGS.data_dir,
        _merged = merged,
        _train_op = train_op,
        _loss = loss,
        _global_step = global_step,
        _feed_X = images,
        _feed_Y = labels,
        _train_X = train_X,
        _train_Y = train_Y,
        _is_training_ph = is_training_ph,
        _top_k_op = top_1_op,
        _lr = lr)
    print('---------------------register is OK-------------------')
    strain_wk.run(_simulate = FLAGS.sleep_time)	
    sess.close()

def process_server(device):
    global sess, merged, global_step, images, labels, is_training_ph, train_X, train_Y,train_X2, train_Y2 
    global lr, train_op, eval_ops, loss, loss_ps, top_1_op,top_5_op,top_1_op_ps, top_5_op_ps
    strain_ps = st_ps.STRAIN_PS(
        _total_worker_num = FLAGS.worker_num,
        _check_period = FLAGS.check_period, 
        _class_num = FLAGS.class_num,
        _base_dir = FLAGS.base_dir,
        _host = FLAGS.host,
        _port_base = FLAGS.port_base,
        _band_width_limit = FLAGS.band_width_limit,
        _training_end = FLAGS.training_end,
        _epsilon = FLAGS.epsilon,
        _batch_size = FLAGS.batch_size, 
        _s = FLAGS.s,
        _buffsize = FLAGS.BUFF_SIZE)
    build_model(device)
    print('---------------------build_model is OK-------------------')
    strain_ps.register(
        _session = sess,
        _data_dir = FLAGS.data_dir,
        _merged = merged,
        _train_op = train_op,
        _loss = loss,
        _global_step = global_step,
        _feed_X = images,
        _feed_Y = labels,
        _train_X = train_X2,
        _train_Y = train_Y2,
        _is_training_ph = is_training_ph,
        _top_1_op = top_1_op,
        _top_5_op = top_5_op,
        _lr = lr)
    print('---------------------register is OK-------------------')
    strain_ps.run()
def main(argv):
    cuda_count_number = 0
    if(FLAGS.job_name == 'ps'):
        device = '/cpu:0'
        process_server(device)
    elif(FLAGS.job_name == 'worker'):
        device = ('/gpu:%d' % (FLAGS.worker_index+cuda_count_number))
        process_worker(device)
    else:
        print("ArgumentError:argument <job_name> must be ps or worker")

if __name__ == "__main__":
  tf.app.run()


