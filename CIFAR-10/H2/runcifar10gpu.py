
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import load_cifar10
import numpy as np
import tensorflow as tf

import os
import sys
import tarfile

import re

from six.moves import xrange 
#from six.moves import urllib
#from resnet import *

#from resnet import inference_small

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1" 

config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.33  
config.gpu_options.allow_growth = True   
sess = tf.Session(config = config)

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
flags.DEFINE_string('data_dir', '/home/zjlab/yangduo/cifar10_data', """Path to the CIFAR-10 data directory.""")
flags.DEFINE_integer('BUFF_SIZE', 1024, """Number of images to process in a batch.""")                      
# both worker and ps
flags.DEFINE_string('base_dir', 'wbsp/', 'The path where log info will be stored')
#/lidaox/yangduo/TensorFlow/STrain-main/ours_cifar10/base/    
flags.DEFINE_string('host', 'localhost', 'IP address of parameter server')
flags.DEFINE_integer('port_base', 2222, 'Start port for listening to workers')
flags.DEFINE_string('job_name', '', 'Either ps or worker')
flags.DEFINE_integer('class_num', 10, 'Training batch size ')
flags.DEFINE_float('check_period', 5.0, 'Length of time between two checkpoints')

# wk
flags.DEFINE_integer('worker_index', 0, 'Index of worker')
flags.DEFINE_float('base_time_step', 1.0, 'Initial length of time between two commits for each worker')
flags.DEFINE_integer('max_steps', 1000000, 'Max steps each worker is allowed to run')
flags.DEFINE_float('sleep_time', 1.0, 'Specify the sleep_time')
# ps
flags.DEFINE_integer('worker_num', 2, 'Total number of workers')
flags.DEFINE_float('band_width_limit', None, 'Specify the constrait of bandwidth in the form of x M/s')
flags.DEFINE_float('training_end', 0.1, 'When loss is smaller than this, end training')
flags.DEFINE_float('epsilon', 0.3, 'Epsilon')
FLAGS = flags.FLAGS
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
# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

#if tf.gfile.Exists(FLAGS.base_dir):
#	tf.gfile.DeleteRecursively(FLAGS.base_dir)
#tf.gfile.MakeDirs(FLAGS.base_dir)

NUM_EPOCHS_PER_DECAY = 10.0      # Epochs after which learning rate decays.

def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size
def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
  #with tf.device('/gpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inference_cnn(images):
  #Build the CIFAR-10 model.
  #Args:images: Images returned from distorted_inputs() or inputs(). [None, 32, 32, 3]
  #Returns: Logits.
  # We instantiate all variables using tf.get_variable() instead of tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function by replacing all instances of tf.get_variable() with tf.Variable().
  # conv1 - [None, 32, 32, 64]
  with tf.variable_scope('conv1') as scope:
    weights1 = _variable_with_weight_decay('weights1', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.004)
    conv = tf.nn.conv2d(images, weights1, [1, 1, 1, 1], padding='SAME')
    biases1 = _variable_on_cpu('biases1', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases1)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1) 
  # pool1 - [None, 16, 16, 64]
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
  # norm1 - [None, 16, 16, 64]
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
  # conv2 - [None, 16, 16, 64]
  with tf.variable_scope('conv2') as scope:
    weights2 = _variable_with_weight_decay('weights2', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.004)
    conv = tf.nn.conv2d(norm1, weights2, [1, 1, 1, 1], padding='SAME')
    biases2 = _variable_on_cpu('biases2', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases2)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)
  # norm2 - [None, 16, 16, 64]
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2 - [None, 8, 8, 64]
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  with tf.variable_scope('softmax_linear') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    # dim = reshape.get_shape()[1].value
    dim = 8 * 8 * 64
    reshape = tf.reshape(pool2, [-1, dim])            #[4096,10]
    weights3 = _variable_with_weight_decay('weights3', [dim, 10], stddev=1/192.0, wd=0.004)
    biases3 = _variable_on_cpu('biases3', [10], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(reshape, weights3), biases3, name=scope.name)
    _activation_summary(softmax_linear)
  return softmax_linear

def build_model():
    #global sess, fetch_data, merged, train_op, loss, global_step, images, labels, top_k_op
    global sess, merged, train_op, loss, cross_entropy, global_step, images, labels, train_X, train_Y,train_X2, train_Y2,top_1_op,top_5_op
    global lr    
    # build the model
    """Train CIFAR-10 for a number of steps."""
    # tf.Graph().as_default()	
    global_step = tf.train.get_or_create_global_step()
    tf.summary.scalar('global_step', global_step)

    images = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
    labels = tf.placeholder(dtype=tf.int32, shape=[None])
    
    train_X, train_Y = load_cifar10.distorted_inputs(FLAGS.data_dir, FLAGS.batch_size)
    train_X2, train_Y2 = load_cifar10.inputs(True, FLAGS.data_dir, FLAGS.batch_size)
    # # resnet50 ----------------------------------------
    # logits = inference_small(images,
                             # num_classes=10,
                             # is_training=True,
                             # use_bias=(not True),
                             # num_blocks=3)
    # cnn ----------------------------------------
    logits = inference_cnn(images)
    # loss---------------------------------------
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([cross_entropy_mean] + regu_losses)
    #--------------------------------------------
    predictions = tf.nn.softmax(logits)
    top_1_op = top_k_error(predictions, labels, 1)
    top_5_op = top_k_error(predictions, labels, 5)
    # loss_avg
    ema = tf.train.ExponentialMovingAverage(0.9, global_step)
    #tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss]))
    tf.add_to_collection('resnet_update_ops', ema.apply([loss]))
    
    decay_steps = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size * NUM_EPOCHS_PER_DECAY) # 50000/128 * 10 = 390.625
    #lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps,
    #                              LEARNING_RATE_DECAY_FACTOR, staircase=True) #每一步都改变lr；True 每隔 decay_steps 步改变一次
    
    #boundaries = [4 * decay_steps, 8 * decay_steps]
    #boundaries = [round(0.5 * decay_steps), 1 * decay_steps]
    boundaries = [round(2 * decay_steps), round(3.5 * decay_steps)]
    learing_rates = [0.1, 0.01, 0.001]
    lr = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learing_rates)
    # #opt = tf.train.MomentumOptimizer(0.01, 0.9)
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    #batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates = tf.get_collection('resnet_update_ops')
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)
    #train_op = opt.minimize(loss, global_step=global_step)
    
    saver = tf.train.Saver(tf.all_variables())
    
    init = tf.global_variables_initializer()
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()	
    #sess = tf.Session()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    sess.run(init)
    
    # ckpt_dir = os.path.join(FLAGS.base_dir + 'ckpt_%d/' % (FLAGS.worker_index))
    # ckpt = tf.train.latest_checkpoint(ckpt_dir)
    # if ckpt != None: #如果有断点文件，读取最近的断点文件
        # saver.restore(sess,ckpt)
    
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess = sess, coord = coord)

def process_worker():
    #global sess, fetch_data, merged, train_op, loss, global_step, images, labels, top_k_op
    global sess, merged, train_op, loss, cross_entropy, global_step, images, labels, train_X, train_Y, top_1_op,top_5_op
    global lr
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
            #_s = FLAGS.s,
            _buffsize = FLAGS.BUFF_SIZE)
    #print('---------------------STRAIN_WK is OK-------------------')
    # cross_entropy is only related to loss_cnt, maybe it can be deleted: huhanpeng
    build_model()
    print('---------------------build_model is OK-------------------')
    strain_wk.register(
        _session = sess,
        #_func_fetch_data = fetch_data,
        _data_dir = FLAGS.data_dir,
        _merged = merged,
        _train_op = train_op,
        _loss = loss,
        #_loss = cross_entropy,
        _global_step = global_step,
        _feed_X = images,
        _feed_Y = labels,
        _train_X = train_X,
        _train_Y = train_Y,
        _top_k_op = top_1_op,
        _lr = lr)
    print('---------------------register is OK-------------------')
    strain_wk.run(_simulate = FLAGS.sleep_time)	
    sess.close()

def process_server():
    #global sess, fetch_data, merged, train_op, loss, global_step, images, labels, top_k_op
    global sess, merged, train_op, loss, cross_entropy, global_step, images, labels, train_X2, train_Y2, top_1_op,top_5_op
    global lr
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
    #print('---------------------STRAIN_PS is OK-------------------')
    build_model()
    print('---------------------build_model is OK-------------------')
    strain_ps.register(
        _session = sess,
        #_func_fetch_data = fetch_data,
        _data_dir = FLAGS.data_dir,
        _merged = merged,
        _train_op = train_op,
        _loss = loss,
        #_loss = cross_entropy,
        _global_step = global_step,
        _feed_X = images,
        _feed_Y = labels,
        _train_X = train_X2,
        _train_Y = train_Y2,
        _top_1_op = top_1_op,
        _top_5_op = top_5_op,
        _lr = lr)
    print('---------------------register is OK-------------------')
    strain_ps.run()
def main(argv):
    
    if(FLAGS.job_name == 'ps'):
        process_server()
    elif(FLAGS.job_name == 'worker'):
        process_worker()
    else:
        print("ArgumentError:argument <job_name> must be ps or worker")

if __name__ == "__main__":
  tf.app.run()


