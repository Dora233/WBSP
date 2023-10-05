# -*-coding: utf-8 -*-
"""
    @Project: create_tfrecord
    @File   : create_tfrecord.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-07-27 17:19:54
    @desc   : ½«Í¼Æ¬Êý¾Ý±£´æÎªµ¥¸ötfrecordÎÄ¼þ
"""

##########################################################################

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image


##########################################################################
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# Éú³É×Ö·û´®ÐÍµÄÊôÐÔ
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# Éú³ÉÊµÊýÐÍµÄÊôÐÔ
def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def get_example_nums(tf_records_filenames):
    '''
    Í³¼Ætf_recordsÍ¼ÏñµÄ¸öÊý(example)¸öÊý
    :param tf_records_filenames: tf_recordsÎÄ¼þÂ·¾¶
    :return:
    '''
    nums= 0
    for record in tf.python_io.tf_record_iterator(tf_records_filenames):
        nums += 1
    return nums

def show_image(title,image):
    '''
    ÏÔÊ¾Í¼Æ¬
    :param title: Í¼Ïñ±êÌâ
    :param image: Í¼ÏñµÄÊý¾Ý
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    plt.imshow(image)
    plt.axis('on')    # ¹Øµô×ø±êÖáÎª off
    plt.title(title)  # Í¼ÏñÌâÄ¿
    plt.show()

def load_labels_file(filename,labels_num=1,shuffle=False):
    '''
    ÔØÍ¼txtÎÄ¼þ£¬ÎÄ¼þÖÐÃ¿ÐÐÎªÒ»¸öÍ¼Æ¬ÐÅÏ¢£¬ÇÒÒÔ¿Õ¸ñ¸ô¿ª£ºÍ¼ÏñÂ·¾¶ ±êÇ©1 ±êÇ©2£¬Èç£ºtest_image/1.jpg 0 2
    :param filename:
    :param labels_num :labels¸öÊý
    :param shuffle :ÊÇ·ñ´òÂÒË³Ðò
    :return:images type->list
    :return:labels type->list
    '''
    images=[]
    labels=[]
    with open(filename) as f:
        lines_list=f.readlines()
        if shuffle:
            random.shuffle(lines_list)

        for lines in lines_list:
            line=lines.rstrip().split(' ')
            label=[]
            for i in range(labels_num):
                label.append(int(line[i+1]))
            images.append(line[0])
            labels.append(label)
    return images,labels

def read_image(filename, resize_height, resize_width,normalization=False):
    '''
    ¶ÁÈ¡Í¼Æ¬Êý¾Ý,Ä¬ÈÏ·µ»ØµÄÊÇuint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:ÊÇ·ñ¹éÒ»»¯µ½[0.,1.0]
    :return: ·µ»ØµÄÍ¼Æ¬Êý¾Ý
    '''

    bgr_image = cv2.imread(filename)
    if len(bgr_image.shape)==2:#ÈôÊÇ»Ò¶ÈÍ¼Ôò×ªÎªÈýÍ¨µÀ
        print("Warning:gray image",filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)#½«BGR×ªÎªRGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    if resize_height>0 and resize_width>0:
        rgb_image=cv2.resize(rgb_image,(resize_width,resize_height))
    rgb_image=np.asanyarray(rgb_image)
    if normalization:
        # ²»ÄÜÐ´³É:rgb_image=rgb_image/255
        rgb_image=rgb_image/255.0
    # show_image("src resize image",image)
    return rgb_image


def get_batch_images(images,labels,batch_size,labels_nums,one_hot=False,shuffle=False,num_threads=1):
    '''
    :param images:Í¼Ïñ
    :param labels:±êÇ©
    :param batch_size:
    :param labels_nums:±êÇ©¸öÊý
    :param one_hot:ÊÇ·ñ½«labels×ªÎªone_hotµÄÐÎÊ½
    :param shuffle:ÊÇ·ñ´òÂÒË³Ðò,Ò»°ãtrainÊ±shuffle=True,ÑéÖ¤Ê±shuffle=False
    :return:·µ»ØbatchµÄimagesºÍlabels
    '''
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size  # capacity>=min_after_dequeue
    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images,labels],
                                                                    batch_size=batch_size,
                                                                    capacity=capacity,
                                                                    min_after_dequeue=min_after_dequeue,
                                                                    num_threads=num_threads)
    else:
        images_batch, labels_batch = tf.train.batch([images,labels],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        num_threads=num_threads)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, labels_nums, 1, 0)
    return images_batch,labels_batch

def read_records(filename,resize_height, resize_width,type=None):
    '''
    ½âÎörecordÎÄ¼þ:Ô´ÎÄ¼þµÄÍ¼ÏñÊý¾ÝÊÇRGB,uint8,[0,255],Ò»°ã×÷ÎªÑµÁ·Êý¾ÝÊ±,ÐèÒª¹éÒ»»¯µ½[0,1]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param type:Ñ¡ÔñÍ¼ÏñÊý¾ÝµÄ·µ»ØÀàÐÍ
         None:Ä¬ÈÏ½«uint8-[0,255]×ªÎªfloat32-[0,255]
         normalization:¹éÒ»»¯float32-[0,1]
         centralization:¹éÒ»»¯float32-[0,1],ÔÙ¼õ¾ùÖµÖÐÐÄ»¯
    :return:
    '''
    # ´´½¨ÎÄ¼þ¶ÓÁÐ,²»ÏÞ¶ÁÈ¡µÄÊýÁ¿
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader´ÓÎÄ¼þ¶ÓÁÐÖÐ¶ÁÈëÒ»¸öÐòÁÐ»¯µÄÑù±¾
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # ½âÎö·ûºÅ»¯µÄÑù±¾
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)#»ñµÃÍ¼ÏñÔ­Ê¼µÄÊý¾Ý

    tf_height = features['height']
    tf_width = features['width']
    tf_depth = features['depth']
    tf_label = tf.cast(features['label'], tf.int32)
    # PS:»Ö¸´Ô­Ê¼Í¼ÏñÊý¾Ý,reshapeµÄ´óÐ¡±ØÐëÓë±£´æÖ®Ç°µÄÍ¼ÏñshapeÒ»ÖÂ,·ñÔò³ö´í
    # tf_image=tf.reshape(tf_image, [-1])    # ×ª»»ÎªÐÐÏòÁ¿
    tf_image=tf.reshape(tf_image, [resize_height, resize_width, 3]) # ÉèÖÃÍ¼ÏñµÄÎ¬¶È

    # »Ö¸´Êý¾Ýºó,²Å¿ÉÒÔ¶ÔÍ¼Ïñ½øÐÐresize_images:ÊäÈëuint->Êä³öfloat32
    # tf_image=tf.image.resize_images(tf_image,[224, 224])

    # ´æ´¢µÄÍ¼ÏñÀàÐÍÎªuint8,tensorflowÑµÁ·Ê±Êý¾Ý±ØÐëÊÇtf.float32
    if type is None:
        tf_image = tf.cast(tf_image, tf.float32)
    elif type=='normalization':# [1]ÈôÐèÒª¹éÒ»»¯ÇëÊ¹ÓÃ:
        # ½öµ±ÊäÈëÊý¾ÝÊÇuint8,²Å»á¹éÒ»»¯[0,255]
        # tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255.0)  # ¹éÒ»»¯
    elif type=='centralization':
        # ÈôÐèÒª¹éÒ»»¯,ÇÒÖÐÐÄ»¯,¼ÙÉè¾ùÖµÎª0.5,ÇëÊ¹ÓÃ:
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255) - 0.5 #ÖÐÐÄ»¯

    # ÕâÀï½ö½ö·µ»ØÍ¼ÏñºÍ±êÇ©
    # return tf_image, tf_height,tf_width,tf_depth,tf_label
    return tf_image,tf_label


def create_records(image_dir,file, output_record_dir, resize_height, resize_width,shuffle,log=5):
    '''
    ÊµÏÖ½«Í¼ÏñÔ­Ê¼Êý¾Ý,label,³¤,¿íµÈÐÅÏ¢±£´æÎªrecordÎÄ¼þ
    ×¢Òâ:¶ÁÈ¡µÄÍ¼ÏñÊý¾ÝÄ¬ÈÏÊÇuint8,ÔÙ×ªÎªtfµÄ×Ö·û´®ÐÍBytesList±£´æ,½âÎöÇëÐèÒª¸ù¾ÝÐèÒª×ª»»ÀàÐÍ
    :param image_dir:Ô­Ê¼Í¼ÏñµÄÄ¿Â¼
    :param file:ÊäÈë±£´æÍ¼Æ¬ÐÅÏ¢µÄtxtÎÄ¼þ(image_dir+file¹¹³ÉÍ¼Æ¬µÄÂ·¾¶)
    :param output_record_dir:±£´ærecordÎÄ¼þµÄÂ·¾¶
    :param resize_height:
    :param resize_width:
    PS:µ±resize_height»òÕßresize_width=0ÊÇ,²»Ö´ÐÐresize
    :param shuffle:ÊÇ·ñ´òÂÒË³Ðò
    :param log:logÐÅÏ¢´òÓ¡¼ä¸ô
    '''
    # ¼ÓÔØÎÄ¼þ,½ö»ñÈ¡Ò»¸ölabel
    images_list, labels_list=load_labels_file(file,1,shuffle)

    writer = tf.python_io.TFRecordWriter(output_record_dir)
    for i, [image_name, labels] in enumerate(zip(images_list, labels_list)):
        image_path=os.path.join(image_dir,images_list[i])
        if not os.path.exists(image_path):
            print('Err:no image',image_path)
            continue
        image = read_image(image_path, resize_height, resize_width)
        image_raw = image.tostring()
        if i%log==0 or i==len(images_list)-1:
            print('------------processing:%d-th------------' % (i))
            print('current image_path=%s' % (image_path),'shape:{}'.format(image.shape),'labels:{}'.format(labels))
        # ÕâÀï½ö±£´æÒ»¸ölabel,¶àlabelÊÊµ±Ôö¼Ó"'label': _int64_feature(label)"Ïî
        label=labels[0]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()

def disp_records(record_file,resize_height, resize_width,show_nums=4):
    '''
    ½âÎörecordÎÄ¼þ£¬²¢ÏÔÊ¾show_numsÕÅÍ¼Æ¬£¬Ö÷ÒªÓÃÓÚÑéÖ¤Éú³ÉrecordÎÄ¼þÊÇ·ñ³É¹¦
    :param tfrecord_file: recordÎÄ¼þÂ·¾¶
    :return:
    '''
    # ¶ÁÈ¡recordº¯Êý
    tf_image, tf_label = read_records(record_file,resize_height,resize_width,type='normalization')
    # ÏÔÊ¾Ç°4¸öÍ¼Æ¬
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(show_nums):
            image,label = sess.run([tf_image,tf_label])  # ÔÚ»á»°ÖÐÈ¡³öimageºÍlabel
            # image = tf_image.eval()
            # Ö±½Ó´Órecord½âÎöµÄimageÊÇÒ»¸öÏòÁ¿,ÐèÒªreshapeÏÔÊ¾
            # image = image.reshape([height,width,depth])
            print('shape:{},tpye:{},labels:{}'.format(image.shape,image.dtype,label))
            # pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            # pilimg.show()
            show_image("image:%d"%(label),image)
        coord.request_stop()
        coord.join(threads)


def batch_test(record_file,resize_height, resize_width):
    '''
    :param record_file: recordÎÄ¼þÂ·¾¶
    :param resize_height:
    :param resize_width:
    :return:
    :PS:image_batch, label_batchÒ»°ã×÷ÎªÍøÂçµÄÊäÈë
    '''
    # ¶ÁÈ¡recordº¯Êý
    tf_image,tf_label = read_records(record_file,resize_height,resize_width,type='normalization')
    image_batch, label_batch= get_batch_images(tf_image,tf_label,batch_size=4,labels_nums=5,one_hot=False,shuffle=False)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:  # ¿ªÊ¼Ò»¸ö»á»°
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(4):
            # ÔÚ»á»°ÖÐÈ¡³öimagesºÍlabels
            images, labels = sess.run([image_batch, label_batch])
            # ÕâÀï½öÏÔÊ¾Ã¿¸öbatchÀïµÚÒ»ÕÅÍ¼Æ¬
            show_image("image", images[0, :, :, :])
            print('shape:{},tpye:{},labels:{}'.format(images.shape,images.dtype,labels))

        # Í£Ö¹ËùÓÐÏß³Ì
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # ²ÎÊýÉèÖÃ

    resize_height = 299  # Ö¸¶¨´æ´¢Í¼Æ¬¸ß¶È
    resize_width = 299  # Ö¸¶¨´æ´¢Í¼Æ¬¿í¶È
    shuffle=True
    log=5
    # ²úÉútrain.recordÎÄ¼þ
    image_dir='/lidaox/yangduo/ImageNet/ILSVRC2012/train_imagenet'
    train_labels = '/lidaox/zzc/NetPlacer/inception/dataset/train.txt'  # Í¼Æ¬Â·¾¶
    train_record_output = '/lidaox/zzc/NetPlacer/inception/dataset/record/train{}.tfrecords'.format(resize_height)
    create_records(image_dir,train_labels, train_record_output, resize_height, resize_width,shuffle,log)
    train_nums=get_example_nums(train_record_output)
    print("save train example nums={}".format(train_nums))

    # ²úÉúval.recordÎÄ¼þ
    image_dir='/lidaox/yangduo/ImageNet/ILSVRC2012/val'
    val_labels = '/lidaox/zzc/NetPlacer/inception/dataset/val.txt'  # Í¼Æ¬Â·¾¶
    val_record_output = '/lidaox/zzc/NetPlacer/inception/dataset/record/val{}.tfrecords'.format(resize_height)
    create_records(image_dir,val_labels, val_record_output, resize_height, resize_width,shuffle,log)
    val_nums=get_example_nums(val_record_output)
    print("save val example nums={}".format(val_nums))

    # ²âÊÔÏÔÊ¾º¯Êý
    # disp_records(train_record_output,resize_height, resize_width)
    batch_test(train_record_output,resize_height, resize_width)