from tensorflow.python import *   # tensorflow version 1.10 above 
import numpy as np 
import random
import tensorflow as tf 
import os
import copy
from socket import *
import time
import sys   
# sys.getsizeof()
#import load_cifar10
#--------------------------
import threading
from threading import Thread
mutex = threading.Lock()
#import logging
#import math
#from tensorflow.contrib import *

#MIN_TIME_TO_COMMIT = 10 # avoid frequent commit
#COMMUNICATION_SLEEP = 0
#min_c = 0

class STRAIN_WK(object):
    def __init__(self, 
        _worker_index = None, 
        _check_period = None, 
        _init_base_time_step = None, 
        _max_steps = None, 
        _batch_size = None, 
        _class_num = None, # must be given
        _base_dir = None,
        _host = 'localhost',
        _port_base = None,
        #_s = None,
        _buffsize = None):

        self.No = int(_worker_index)
        self.check_period = float(_check_period)
        self.base_time_step = float(_init_base_time_step) # initial , commit update per 40s
        self.max_steps = int(_max_steps)
        self.batch_size = int(_batch_size)
        self.class_num = int(_class_num) #
        self.base_dir = _base_dir
        self.host = _host
        self.port_base = int(_port_base)
        #self.threshold = _s
        #self.eval_step = _eval_step
        self.buffsize = _buffsize #1024
        
        # log for the worker
        self.f_log = open(os.path.join(self.base_dir + 'wk_loss_%d_wbsp.txt' % (self.No)), 'w')
        self.f_time = open(os.path.join(self.base_dir + 'wk_time_%d_wbsp.txt' % (self.No)), 'w')
        #self.f_time2 = open(os.path.join(self.base_dir + 'wk_time2_%d_osp.txt' % (self.No)), 'w')
        #self.f_pre = open(os.path.join(self.base_dir + 'wk_%d_ssp_pred.txt' % (self.No)), 'w')
        
        
        self.parameter = []  # a list of parameters, parameters are np.array
        self.para_shape = []
        self.wk_variables = []
        
        
        self.time_1 = 0
        self.time_2 = 0
        self.time_3 = 0
        self.push_time = 0
        self.pull_time = 0
        self.push_num = 0
        self.pull_num = 0
        
        
        self.time_push = 0
        self.size_push = 0
        self.time_send_loss_and_step = 0
        self.size_send_loss_and_step = 0
        self.time_recv = 0
        self.size_recv = 0
        self.time_load = 0
        
        self.wk_ep = 0
        self.ep = 0
        self.cur_step = 0
        
        global isoktopull,isoktoload
        isoktopull = 0
        isoktoload = 0
    def sendmsg(self, s):
        return self.skt.sendmsg(s)
    def recvmsg(self):
        return self.skt.recvmsg() 
    def register(self, 
        _session = None,
        _data_dir = None,
        _merged = None,
        _train_op = None,
        _loss = None,
        _global_step = None,
        _feed_X = None,
        _feed_Y = None,
        _train_X = None,
        _train_Y = None,
        _is_training_ph = None,
        _feed_prediction = None,
        _top_k_op = None,
        _lr = None,
        _beta = None
        ):
        self.data_dir = _data_dir
        
        self.sess = _session
        #! About the model
        self.merged =  _merged
        self.train_op = _train_op
        self.loss = _loss
        self.global_step = _global_step
        self.images = _feed_X
        self.labels = _feed_Y
        self.train_X = _train_X
        self.train_Y = _train_Y
        self.is_training_ph = _is_training_ph
        self.feed_prediction = _feed_prediction # addional input
        self.top_k_op = _top_k_op
        self.lr = _lr
        #! connect to the ps
        train_X_batch, train_Y_batch = self.sess.run([self.train_X, self.train_Y])
        self.connect2PS()
    def connect2PS(self):
        # connect to the PS
        #print('buffsize:',self.buffsize) #1024
        self.skt = Communicator(self.host, self.port_base + self.No, 'wk', self.buffsize)
        msg = str(self.skt.recv(), encoding = "utf-8").split(',')
        print('---------------------------------------------------------')
        print('msg1',msg) #['1620733822.052202', ' 1620733826.477792']
        self.timer_from_PS = float(msg[1])
        #self.time_align = float(msg[1]) - float(msg[0])  
        print('worker_index:%d' % (self.No))
        #print('time before:',time.time()-self.timer_from_PS)
        self.wk_variables = tf.trainable_variables()
        #print('time after:',time.time()-self.timer_from_PS)
            
        self.skt.sendall(str("Recv start_t").encode())
        msg = self.skt.recv(self.buffsize) 
        print('msg--connect',msg)# b'OK'
        if(str('OK').encode() in msg):
            # if the worker is the first worker to connect PS, upload the parameters
            #print('variables',tf.trainable_variables()) # 两个卷积层的w b还有softmax层的w b，一共6个
            #print('tf.trainable_variables()--connect',tf.trainable_variables())
            # [<tf.Variable 'conv1/weights1:0' shape=(5, 5, 3, 64) dtype=float32_ref>, <tf.Variable 'conv1/biases1:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'conv2/weights2:0' shape=(5, 5, 64, 64) dtype=float32_ref>, <tf.Variable 'conv2/biases2:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'softmax_linear/weights3:0' shape=(4096, 10) dtype=float32_ref>, <tf.Variable 'softmax_linear/biases3:0' shape=(10,) dtype=float32_ref>]
            for v in tf.trainable_variables():
                index_v = tf.trainable_variables().index(v)
                tmp = v.eval(session = self.sess)
                self.parameter.append(tmp)
                shape = v.get_shape()# shape=tmp.shape
                self.para_shape.append(shape)	
                msg = np.array(tmp).tobytes()#!!!!!!!!!!!!!!!!!!!!!!!!!!!
                self.sendmsg(msg)
                print('Communication size: %fB' % (sys.getsizeof(msg)))
                print('Storage size of tmp: %fB' % (sys.getsizeof(tmp)))
                print('Storage size of param[-1]: %fB' % (sys.getsizeof(self.parameter[-1])))
                print("Shape: %s" % (str(shape)))
                print('time for v:',time.time()-self.timer_from_PS)
            self.sendmsg(str('$').encode())
        else:
            # if the worker is not the first worker to connect PS, load the parameters
            self.skt.sendall(str('Load').encode())
            # Receive  theoverall parameters from the PS
            #print('tf.trainable_variables()--connect',tf.trainable_variables())
            for v in tf.trainable_variables():
                msg = self.recvmsg()
                tmp = np.copy(np.frombuffer(msg, dtype=np.float32))
                shape = v.get_shape()
                self.para_shape.append(shape)
                self.parameter.append(tmp.reshape(shape))
                print('Communication size: %fB' % (sys.getsizeof(msg)))
                print('Storage size: %fB' % (sys.getsizeof(tmp)))
                print("Shape: %s" % (str(shape)))
                print('time for v:',time.time()-self.timer_from_PS)
            self.skt.recv(self.buffsize)
            # update the local mode
            index = 0
            for v in tf.trainable_variables():
                v.load(self.parameter[index], self.sess)
                index += 1
        print('time:',time.time()-self.timer_from_PS)
        print("Worker %d successfully connects to the parameter server [%s:%d]" % (self.No, self.host, (self.port_base + self.No)))   

    def compute_grad(self):
        global isoktopull,isoktoload
        print('ps-------------------------------------------------',self.timer_from_PS)  # 1628822072.156423
        print('timefor1',time.time()-self.timer_from_PS)
        
        self.ep = 0
        #all_data, all_labels = load_cifar10.prepare_train_data(padding_size=2)
        while (self.ep< self.max_steps): #1 000 000
            while(isoktopull==0):
                # if(isoktoload==1):
                    # #mutex.acquire()
                    # tmp_time4 = time.time()
                    # index = 0
                    # for v in self.wk_variables:
                        # v.load(self.parameter[index], self.sess)
                        # index += 1
                    # self.time_load = time.time() - tmp_time4
                    # print("Pulltime4: load params: %f" % (self.time_load))
                    # isoktoload=0
                    # print('%10f:compute_grad:isoktopull = 0'%(time.time() - self.timer_from_PS))
                    # print("WBSP-PCC_WK%d: Pull %d finished" % (self.No, self.pull_num))
                    # #mutex.release()
                #self.ep+= 1
                self.time_1 = time.time() - self.timer_from_PS
                #time_start_2 = time.time()
                print('----------------ep%d starts--------timecompu0:%f--------------'% (self.ep,time.time() - self.timer_from_PS))
                #si += 1
                # train_X, train_Y = load_cifar10.generate_augment_train_batch(
                                    # all_data, all_labels, self.batch_size)
                train_X_batch, train_Y_batch = self.sess.run([self.train_X, self.train_Y])
                #print('timecompu1',time.time() - self.timer_from_PS)
                # eval_X, eval_Y = load_cifar10.generate_test_batch(
                                    # vali_data, vali_labels, 250)
                print('time_train1',time.time()-self.timer_from_PS - self.time_1)
                if(self.No == 0):
                    time.sleep(0)
                if(self.No == 1):
                    time.sleep(200)
                if(self.No == 2):
                    time.sleep(0)
                if(self.No == 3):
                    time.sleep(200)
                if(self.No == 4):
                    time.sleep(0)
                if(self.No == 5):
                    time.sleep(200)
                if(self.No == 6):
                    time.sleep(0)
                if(self.No == 7):
                    time.sleep(200)
                
                print('time_train2',time.time()-self.timer_from_PS - self.time_1)
                _, cur_loss, learn_rate, self.cur_step = self.sess.run([self.train_op, self.loss, self.lr, self.global_step], 
                    feed_dict = {self.images: train_X_batch, self.labels: train_Y_batch, self.is_training_ph: True})
                print('time_train3',time.time()-self.timer_from_PS - self.time_1)
                self.time_2 = time.time() - self.timer_from_PS
                self.f_log.write('%020.10f, %d, %d, %.20f, %.20f\n' % (self.time_2,self.ep, self.cur_step, cur_loss,self.time_2-self.time_1))
                print('cur_time: %020.10f, ep: %d, cur_step: %d, learn_rate: %f, cur_loss: %f\n' % (self.time_2,self.ep, self.cur_step, learn_rate, cur_loss))
                self.f_log.flush()
                #print('tf.trainable_variables()--compu',tf.trainable_variables())   #[]
                self.ep+= 1
                if(isoktopull==1):
                    print('isoktopull',isoktopull)
                    break
            
                self.f_time.write('%.10f, %.20f, %d, %.20f, %d, %.20f, %d, %.20f, %.20f, %.20f\n' % \
                    (time.time() - self.timer_from_PS, self.time_2-self.time_1, self.ep, \
                    self.push_time, self.push_num, self.pull_time,self.pull_num, \
                    self.size_push, self.size_send_loss_and_step, self.size_recv))
                self.f_time.flush() 
                print('---------------------------------------------------------')
            # if(isoktopull3 == 1):
                # print('Time4: ready to update:',time.time()- self.timer_from_PS)
                # tmp_time4 = time.time()
                # index = 0
                # for v in tf.trainable_variables():
                    # v.load(self.parameter[index], self.sess)
                    # index += 1
                # time_load = time.time() - tmp_time4
                # isoktopull3 = 0
                # print("Pulltime4: load params: %f" % (time_load))
        print('WBSP-PCC_WK%d stops training - Step:%d\tTime:%f' % (self.No,self.ep, time.time()-self.timer_from_PS))
        print("WK ends")
        self.f_log.close()
        self.f_time.close()
    
    def push_and_pull(self):
        global isoktopull,isoktopush
        while(True):
            while(isoktopull==0):
                if(self.wk_ep < self.ep):
                    print('ep:%d, wk_ep:%d, cur_step:%d' % (self.ep, self.wk_ep, self.cur_step))
                    print('timenow_1',time.time() - self.timer_from_PS)
                    self.wk_ep += 1
                    tmp1 = time.time() - self.timer_from_PS
                    self.skt.sendall(str('Push').encode())
                    self.skt.recv(self.buffsize)
                    self.wk_process_push()
                    #msg_signal = self.skt.recv(self.buffsize)#EndPush or Pull
                    tmp3 = time.time() - self.timer_from_PS
                    self.push_time += tmp3 - tmp1
                    print('isoktopull_2',isoktopull)
                    if(isoktopull==1):
                        break
            print('timenow_2',time.time() - self.timer_from_PS)    
            tmp4 = time.time() - self.timer_from_PS
            self.skt.sendall(str('OK2Pull').encode())
            self.wk_process_pull()
            isoktopull = 0
            tmp5 = time.time() - self.timer_from_PS
            self.pull_time += tmp5 - tmp4
            print('timenow_3',time.time() - self.timer_from_PS) 
        
    def wk_process_push(self):
        global isoktopull
        print('*********************** push ********************')
        self.push_num += 1
        #print('tf.trainable_variables()--1',tf.trainable_variables())#[]
        #mutex.acquire()
        tmp_time1 = time.time()
        index = 0
        communication_size = 0
        #print('self.wk_variables',self.wk_variables) #[]
        #print('tf.trainable_variables()',tf.trainable_variables())#[]
        for v in self.wk_variables:
            cur_parameter = v.eval(session = self.sess)
            l = (cur_parameter - self.parameter[index])
            communication_size += sys.getsizeof(l)
            #print('index',index)
            #print('l',l.shape)
            msg_l = np.array(l).tobytes()
            self.sendmsg(msg_l)
            self.parameter[index] = cur_parameter[:]
            index += 1
        self.time_push = time.time() - tmp_time1
        print("Pushtime1: extract and send grad: %f" % (self.time_push))
        self.size_push = float(communication_size)/(1024.0 * 1024.0)
        print("The size of transmitted Parameters: %f M" % (self.size_push))
        #mutex.release()
        
        tmp_time2 = time.time()
        send_msg = b'%d, %d,' % (self.ep, self.cur_step)
        self.skt.sendall(send_msg)
        msg1 = self.skt.recv(self.buffsize).split(str(',').encode())
        print('msg1',msg1)
        isoktopull = int(msg1[0])
        print('isoktopull_1',isoktopull)
        self.time_send_loss_and_step = time.time() - tmp_time2
        print("Pushtime2: send msg0-5: %f" % (self.time_send_loss_and_step))
        self.size_send_loss_and_step = float(sys.getsizeof(send_msg))/(1024.0 * 1024.0)
        print("The size of transmitted Parameters: %f M" % (self.size_send_loss_and_step))
        print("WBSP-PCC_WK%d: Push %d finished" % (self.No, self.push_num))
        print('******************************************************')
    
    def wk_process_pull(self):
        #global isoktoload
        print('########################## pull ######################')
        self.pull_num += 1  
        #mutex.acquire()
        communication_size = 0
        tmp_time3 = time.time()
        for i in xrange(len(self.parameter)):
            msg = self.recvmsg()
            communication_size += sys.getsizeof(msg)
            self.parameter[i] = np.copy(np.frombuffer(msg, dtype=np.float32)).reshape(self.para_shape[i])
        self.skt.recv(self.buffsize)#EndPull
        self.time_recv = time.time() - tmp_time3
        #mutex.release()
        print("Pulltime3: recv params: %f" % (self.time_recv))
        self.size_recv = float(communication_size)/(1024.0 * 1024.0)
        print("The size of received Parameters: %f M" % (self.size_recv))
        
        #isoktoload = 1
        #print('%10f:wk_process_pull:isoktoload = 1' %(time.time() - self.timer_from_PS))
        
        tmp_time4 = time.time()
        index = 0
        for v in self.wk_variables:
            v.load(self.parameter[index], self.sess)
            index += 1
        self.time_load = time.time() - tmp_time4
        print("Pulltime4: load params: %f" % (self.time_load))
        print("WBSP-PCC_WK%d: Pull %d finished" % (self.No, self.pull_num))
        print('################################################')
    
    def run(self,_simulate):
        T2 = threading.Thread(target=self.compute_grad, args=())
        T1 = threading.Thread(target=self.push_and_pull, args=())
        T2.start()
        T1.start()
        T2.join()
        T1.join()
        
class Communicator(object):
    def __init__(self, host, port, role, buffsize):
        self.skt = socket(AF_INET,SOCK_STREAM)
        addr = (host, port)
        #print('---------------------------------')
        #print('addr:',addr)
        if role == 'wk':
            self.skt.connect(addr)
            self.buffsize = buffsize
            #print('--------------Communicator-------------------')
            #print('buffsize:',buffsize)
        elif role == 'ps':	
            self.skt.bind(addr)
            self.skt.listen(6)
        else:
            raise ValueError("Role %s is not allowed" % role)

    def sendmsg(self, s): 
        tmp_time = time.time()
        self.skt.sendall(str(len(s)).encode())
        self.skt.recv(self.buffsize)
        self.skt.sendall(s)
        self.skt.recv(self.buffsize)
        return time.time() - tmp_time

    def recvmsg(self):
        s = b''
        length = int(self.skt.recv(self.buffsize))
        self.skt.sendall(str('Start').encode())
        while(length > 0):
            msg = self.skt.recv(self.buffsize)
            s += msg
            length -= len(msg)
        self.skt.sendall(str('OK').encode())
        return s

    def recv(self, buffsize=None):
        buffsize = buffsize if buffsize else self.buffsize
        return self.skt.recv(buffsize)

    def sendall(self, bytes):
        self.skt.sendall(bytes)

    def close():
        self.skt.close()
    
