from tensorflow.python import *   # tensorflow version 1.10 above 
import numpy as np 
import random
import tensorflow as tf 
import os 
import copy
from socket import *
import time
import sys

#import load_cifar10


from scipy.optimize import curve_fit  # for curve fit
import threading
#for ssp
from tensorflow.python.ops import state_ops, variables, variable_scope

START_TIME = time.time()
isfirst = True
oktotest = False
# for listen push pull and get loss
parameters = []
# for listen and push
v_para = []
# for listen and check
mutex = threading.Lock()
# for push
mu = 0.6
eta = 1 / 18.0
MOMENTUN = False
# for push pull check and run
#min_step = 0
# for check and run
checkInCnt = 0
# for LossQueue
WINDOW_LENGHTH_OF_LOSS = 10 # the length of windows for the global loss, used to check whether converge
CONVERGE_VAR = 0.001 # when the variance of loss in WINDOW_LENGHTH_OF_LOSS less equal to CONVERGE_VAR => converge

class PSThread(threading.Thread):	#threading.Thread
    def __init__(self, worker_index, _host, _port_base, _buffsize, _worker_num, _session):
        threading.Thread.__init__(self)
        global local_step_list
        self.index = worker_index
        self.host = _host
        self.port_base = _port_base
        self.buffsize = _buffsize
        #self.threshold = _threshold
        self.worker_num = _worker_num
        self.sess = _session
        
        self.isCheckOrStop = 0	# 0: normal; 1: check, stop; 2: restart; -1: exit
        self.go_on = True	
        self.skt_client = None
        
        # for ssp
        self._const_max_test_step = 10000
        self._last_step = 0
        self._last_time = time.time()
        self._wait_time = 0.01
        #for oursp
        #self.isoktopull = [0 for _ in xrange(worker_num)]
    def sendmsg(self, s):
        length = len(s)
        self.skt_client.sendall(str(length).encode())
        self.skt_client.recv(self.buffsize) # Start
        self.skt_client.sendall(s)
        self.skt_client.recv(self.buffsize) # OK
    def recvmsg(self):
        s = b''
        recv_buff = self.skt_client.recv(self.buffsize)
        length = int(recv_buff.decode())
        self.skt_client.sendall(str('Start').encode())
        while( length > 0):
            msg = self.skt_client.recv(self.buffsize)
            s = s + msg 
            length -= len(msg)
        self.skt_client.sendall(str('OK').encode())
        return s

    # listener starts, listen to the worker
    def run(self):
        self.skt = socket(AF_INET,SOCK_STREAM)		
        addr = (self.host, self.port_base + self.index)     
        self.skt.bind(addr)
        self.skt.listen(3) # the max allowed tcp requre number.# 最大等待数（有很多人理解为最大连接数，其实是错误的）	
        while True:
            self.listen()
        self.skt.close()
    
    # main function for each listener,
    def listen(self):
        global time0_start, time1_push, time2_pull, time3_check
        print('Wait for %dth connection ... ' % self.index)
        self.skt_client, addr_client = self.skt.accept()
        
        print('Connection from :' + str(addr_client))
        print('Listener  %d starts to work' % self.index)
        
        global START_TIME
        self.skt_client.sendall(b'%f, %f' % (START_TIME, time.time())) # send current time to align
        self.skt_client.recv(self.buffsize) # 收到的应该是Recv start_t

        # if this is the first worker, need to initailize parameters
        global isfirst, parameters, oktotest
        mutex.acquire()
        if(isfirst):
            isfirst = False
            self.skt_client.sendall(str('OK').encode()) # a bytes-like object is required, not 'str'
            while True:
                msg = self.recvmsg() # WK 第v个模型参数的值  # WK \u7b2cv\u4e2a\u6a21\u578b\u53c2\u6570\u7684\u503c
                if(b'$' == msg):
                    break
                tmp_l = np.copy(np.frombuffer(msg, dtype=np.float32)) 
                parameters.append(tmp_l)
                v_para.append(np.zeros_like(tmp_l))
                # debug 
                #print('------------------------------Worker %d is first' %(self.index))
                print('Communication size: %fB' % (sys.getsizeof(msg)))
                print('Storage size: %fB' % (sys.getsizeof(tmp_l)))
        else:
            self.skt_client.sendall(str('No').encode())
            self.skt_client.recv(self.buffsize) # Load
            for i in xrange(len(parameters)):
                self.sendmsg(parameters[i].tobytes())
            self.skt_client.sendall(str('$').encode())
        mutex.release()
        
        # compute the size of model
        totalsize = 0.0
        for x in parameters:
            totalsize += sys.getsizeof(x)
        print("The size of model: %fMB" % (totalsize / (1024.0 * 1024.0)))
        oktotest = True
        
        # main loop: start to listen to the port
        global push_num_list, global_step_list, local_step_list
        global isoktopull
        while (True):
            while(isoktopull[self.index]==0):
                #global_step_list[self.index] += 1
                time0_start[self.index] = time.time() - START_TIME
                msg = self.skt_client.recv(self.buffsize)
                print('msg:Push',msg) #Pull
                if(b'Push' in msg):
                    self.skt_client.sendall(str('ready').encode())
                    self.recv_push()
                    time1_push[self.index] = time.time() - START_TIME
                    if(isoktopull[self.index]==1):
                        break
                else:
                    time.sleep(0.000001)
                    print('Error:%s' % msg)
                    
            #self.skt_client.sendall(b'Pull')
            msg_pull = self.skt_client.recv(self.buffsize)
            print('msg:OK2Pull',msg_pull)
            if(b'OK2Pull'in msg_pull):
                self.return_pull()
                time2_pull[self.index] = time.time() - START_TIME
            else:
                print('Error again:%s' % msg_pull)
        print("Worker %d has been totally blocked for %f seconds" % (self.index, total_hung_cnt[self.index]))

    def recv_push(self):
        global local_step_list, worker_step_list, worker_loss_list
        global parameters, worker_num, mu, eta, MOMENTUN
        global push_time,push_size,isoktopull,isoktopull_ps
        #global time_train, time_train_test, time_train_test_push
        print('***********************Worker %d recv push ********************'% (self.index))  
        tmp_time1 = time.time()
        #while True:
        communication_size = 0
        for i in xrange(len(parameters)): # 0:5
            msg = self.recvmsg()
            communication_size += sys.getsizeof(msg)
            new_l = np.copy(np.frombuffer(msg, dtype=np.float32))
            #print('new_l.shape',new_l.shape)
            parameters[i] = parameters[i] + new_l / float(self.worker_num)
        #self.skt_client.sendall(str('over').encode())  
        #print('send over OK')
        len1 = int(local_step_list[1]) # 1代表慢WK
        msg_l = self.skt_client.recv(self.buffsize).split(str(',').encode())
        print('msg_l:',msg_l)
        local_step_list[self.index] = int(msg_l[0]) # ep 原来的
        worker_step_list[self.index] = int(msg_l[1]) # cur_step from self.global_step
        print('local_step_0:%f, local_step_1:%f' % (local_step_list[0],local_step_list[1]))
        len2 = int(local_step_list[1])
        if(len2 > len1):# 如果慢WK有step更新，说明已经完成了新一轮训练，可以更新全局模参了
            isoktopull_ps = [1 for _ in xrange(worker_num)]
            print('isoktopull_a',isoktopull_ps)
        isoktopull[self.index] = isoktopull_ps[self.index]
        self.skt_client.sendall(b'%d, %d, %d,' % (isoktopull[self.index],isoktopull[self.index],isoktopull[self.index]))
        print('send signal to worker %d over'%(self.index))
        push_time[self.index] = time.time() - tmp_time1 
        push_size[self.index] = float(communication_size)/(1024.0 * 1024.0)
        
    def return_pull(self):
        global parameters, pull_time, pull_size
        global isoktopull,isoktopull_ps
        #self.skt_client.sendall(str('isoktopull').encode())   
        #msg = self.skt_client.recv(self.buffsize) # OK
        # print('msg:',msg)
        #l = "ok, %d," % (isoktopull[self.index])
        #self.skt_client.sendall(str(l).encode())
        #if(isoktopull[self.index] == 1):
        
        tmp_time3 = time.time()
        communication_size = 0
        for i in xrange(len(parameters)):
            msg = np.array(parameters[i]).tobytes()
            self.sendmsg(msg)
            communication_size += sys.getsizeof(msg)
        self.skt_client.sendall(str('EndPull').encode())
        pull_time[self.index] = time.time() - tmp_time3
        pull_size[self.index] = float(communication_size)/(1024.0 * 1024.0)
        isoktopull[self.index] = 0
        isoktopull_ps[self.index] = 0
        print('isoktopull_b',isoktopull)
        print('****************************Worker %d return pull****************************' % (self.index))
        
        

class LossQueue(object):
    '''
        we do not need a get function in this implementation of queue
    '''
    def __init__(self, maxsize, training_end):
        self.maxsize = maxsize
        self.train_end = training_end
        self.queue = [0.0] * self.maxsize
        self.start_point = 0
        self.end_point = 0
        self.cur_size = 0
    def put(self, elem):
        '''
            put elem to the queue
            1. if the queue is not full, direct put the elem in
            2. if the queue is full, keep the size the same and put the elem iteratively
        '''
        if(self.cur_size == 0):
            self.queue[self.end_point] = elem
            self.cur_size += 1
        elif(self.cur_size < self.maxsize):
            self.end_point += 1
            self.queue[self.end_point] = elem
            self.cur_size += 1
        elif(self.end_point < self.maxsize - 1):
            self.end_point += 1
            self.queue[self.end_point] = elem
        else:
            self.end_point = 0
            self.queue[self.end_point] = elem
    def isConverge(self):
        if(self.cur_size < self.maxsize):
            return False, None
        var = np.var(np.array(self.queue))
        mean = np.mean(np.array(self.queue))
        #if(var < CONVERGE_VAR): # 0.001 
        if((var < 0.001) & (mean <= 0.7)):
        #if((var < CONVERGE_VAR) & (self.queue[self.end_point] < self.train_end)):
            return True, var
        else: 
            return False, var

class STRAIN_PS(object):
    def __init__(self,
        _total_worker_num = None,
        _check_period = None, 
        _class_num = None,
        _base_dir = None,
        _host = 'localhost',
        _port_base = None,
        _band_width_limit = None,
        _training_end = None,
        _epsilon = None,
        _batch_size = None,
        _s = None,
        _buffsize = None):

        global class_num, worker_num
        worker_num = int(_total_worker_num)
        self.check_period = float(_check_period)
        class_num = int(_class_num)
        self.base_dir = _base_dir
        self.host = _host 
        self.port_base = int(_port_base)
        self.band_width_limit = _band_width_limit
        self.training_end = _training_end
        #_epsilon
        self.batch_size = int(_batch_size)
        #self.threshold = _s
        self.buffsize = _buffsize

        # for run
        self.global_loss = None
        self.trainingEndorNot = False
        self.global_loss_q = LossQueue(maxsize=50, training_end = self.training_end) #WINDOW_LENGHTH_OF_LOSS=10
        #self.isCheckOrStop = 0	# 0: normal; 1: check, stop; 2: restart; -1: exit
        #self.go_on = True
        
        global isoktopull,isoktopull_ps
        isoktopull = [0 for _ in xrange(worker_num)]
        isoktopull_ps = [0 for _ in xrange(worker_num)]
        
        # for record: ps_wkloss_pushnum_ssp
        global local_step_list, worker_step_list, worker_loss_list, push_num_list, global_step_list
        local_step_list = [0 for _ in xrange(worker_num)] #原来的 num_of_local_update
        worker_step_list = [0 for _ in xrange(worker_num)]
        worker_loss_list = [0.0 for _ in xrange(worker_num)]
        push_num_list = [0 for _ in xrange(worker_num)] #原来的 commit_cnt_list
        global_step_list = [0 for _ in xrange(worker_num)]
        # for record: ps_time_ssp
        global push_totaltime_list, total_hung_cnt
        #, blocked_time_list
        global time0_start, time1_push, time2_pull, time3_check, push_time, hung_cnt, pull_time
        global push_size, pull_size
        push_totaltime_list = [0.0 for _ in xrange(worker_num)]
        #blocked_time_list = [0.0 for _ in xrange(worker_num)]
        total_hung_cnt = [0.0 for _ in xrange(worker_num)]
        time0_start = [0.0 for _ in xrange(worker_num)] 
        time1_push = [0.0 for _ in xrange(worker_num)]
        time2_pull = [0.0 for _ in xrange(worker_num)]
        time3_check = [0.0 for _ in xrange(worker_num)]
        push_time = [0.0 for _ in xrange(worker_num)]
        hung_cnt = [0.0 for _ in xrange(worker_num)]
        pull_time = [0.0 for _ in xrange(worker_num)]
        push_size = [0.0 for _ in xrange(worker_num)]
        pull_size = [0.0 for _ in xrange(worker_num)]
        # open name
        #self.f_wkloss_pushnum = open(os.path.join(self.base_dir + 'ps_wkloss_pushnum_osp.txt'), 'w')
        self.f_time = open(os.path.join(self.base_dir + 'ps_time_wbsp.txt'), 'w')
        #self.f_time2 = open(os.path.join(self.base_dir + 'ps_time2_osp.txt'), 'w')
        self.f_global_loss = open(os.path.join(self.base_dir + 'ps_global_loss_wbsp.txt'), 'w')
        #self.f_global_eval = open(os.path.join(self.base_dir + 'ps_global_eval_osp.txt'), 'w')
        
        # for prediction and evaluation
        self.predict_cnt = [0 for _ in xrange(class_num)]
        self.predict_rst = [0 for _ in xrange(class_num)]
        self.global_eval_rst = [0.0 for _ in xrange(class_num + 1)]
        	
    def register(self, 
        _session = None, 
        #_func_fetch_data = None,
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
        _top_1_op = None,
        _top_5_op = None,
        _lr = None,
        _beta = None
        ):
        #self.func_fetch_data = _func_fetch_data
        self.sess = _session
        self.data_dir = _data_dir
        self.merged =  _merged
        self.train_op = _train_op
        self.loss = _loss
        self.global_step = _global_step
        self.images = _feed_X
        self.labels = _feed_Y
        self.train_X = _train_X
        self.train_Y = _train_Y
        self.is_training_ph = _is_training_ph
        self.feed_prediction = _feed_prediction
        self.top_1_op = _top_1_op
        self.top_5_op = _top_5_op
        self.lr = _lr
        # _beta

    def get_global_loss(self):
        #self.loadmodel()
        global parameters, push_time,push_size, pull_time, pull_size
        index = 0
        for v in tf.trainable_variables():
            shape = v.get_shape()
            load_v = np.array(parameters[index]).reshape(shape)
            v.load(load_v, self.sess)
            index += 1
        
        #self.evaluation(10)
        global class_num, worker_step_list, START_TIME
        cur_time = time.time() - START_TIME
        self.global_loss = 0.0
        self.predictions_1 = 0.0
        self.predictions_5 = 0.0
        eval_time = 10
        #all_eval_images, all_eval_labels = load_cifar10.prepare_eval_data(padding_size=2) # (10000, 32, 32, 3)
        
        for _ in xrange(eval_time):
            #eval_images, eval_labels = load_cifar10.generate_augment_eval_batch(
            #                    all_eval_images, all_eval_labels, self.batch_size)
            eval_images, eval_labels = self.sess.run([self.train_X, self.train_Y])
            #print('eval_images.shape',eval_images.shape) # (128, 32, 32, 3)
            #print('eval_labels.shape',eval_labels.shape) # (128,)
            predictions_1,predictions_5, _loss, learn_rate, global_step_ = self.sess.run([self.top_1_op,self.top_5_op, self.loss, self.lr, self.global_step],
                                feed_dict={self.images: eval_images, self.labels: eval_labels, self.is_training_ph: False})
            
            self.global_loss += _loss
            self.predictions_1 += predictions_1
            self.predictions_5 += predictions_5
            #for i in xrange(len(eval_labels)):
            #    self.predict_cnt[int(eval_labels[i])] += 1
            #    if(predictions[i]): #TypeError: list indices must be integers or slices, not numpy.float64
            #        self.predict_rst[int(eval_labels[i])] += 1
        self.global_loss = self.global_loss / float(eval_time)
        self.predictions_1 = self.predictions_1 / float(eval_time)
        self.predictions_5 = self.predictions_5 / float(eval_time)
        print("Global loss: %f Total top 1 accuracy: %.2f Total top 5 accuracy: %.2f" % (self.global_loss, self.predictions_1, self.predictions_5))
            #(self.global_loss, sum(self.predict_rst) / float(sum(self.predict_cnt))))
        
        ## calculate accuracy for each class
        #self.global_eval_rst[-1] = sum(self.predict_rst) / float(sum(self.predict_cnt))
        #for i in xrange(class_num):
        #    if(self.predict_cnt[i] == 0):
        #        self.global_eval_rst[i] = -1.0
        #    else:
        #        self.global_eval_rst[i] = float(self.predict_rst[i]) / self.predict_cnt[i]
        #print('global_eval_rst',self.global_eval_rst) 
        #self.f_global_eval.write('%020.10f: %s\n' % (cur_time, str(self.global_eval_rst)))
        #self.f_global_loss.write('%020.10f, %020.10f, %020.10f, %030.20f\n' % (cur_time, average(worker_step_list), sum(worker_step_list), self.global_loss))
        self.f_global_loss.write('%020.10f, %10d, %10d, %.20f, %.20f, %.20f, %.10f, %.10f, %.10f, %.10f\n' % \
            (cur_time, worker_step_list[1], sum(worker_step_list), self.global_loss, self.predictions_1, self.predictions_5, sum(push_time), sum(push_size), sum(pull_time), sum(pull_size)))
        #self.f_global_eval.flush()
        self.f_global_loss.flush()

        self.predict_cnt = [0 for _ in xrange(class_num)]
        self.predict_rst = [0 for _ in xrange(class_num)]
        
        # converge
        self.global_loss_q.put(self.global_loss)
        isEnd, var = self.global_loss_q.isConverge()
        print('Current variance: %f' % (var if var else 0))
        #if(isEnd):
        if(float(sum(worker_step_list)) >= 1280*200/128*1000000):
            self.trainingEndorNot = True
    
    def record_info(self):
        global START_TIME
        cur_time = time.time() - START_TIME
        global local_step_list, worker_step_list, worker_loss_list, push_num_list, global_step_list
        global push_totaltime_list, total_hung_cnt
        #, blocked_time_list
        global time0_start, time1_push, time2_pull, time3_check, push_time, hung_cnt, pull_time
        
        self.f_time.write('%020.10f: %s, %s, %s, %s, %s, %s, %s, %s\n' % (cur_time, str(time0_start), str(push_time), \
            str(time1_push), str(hung_cnt), str(total_hung_cnt), str(pull_time), str(time2_pull), str(time3_check)))
        self.f_time.flush()
        #self.f_time2.flush()
	# main run of the ps scheduler
    def run(self):
        global worker_num, class_num
        #global expect_commit
        global local_step_list #min_step, 
        # create listeners and launch them
        ps_t = [PSThread(i, self.host, self.port_base, self.buffsize, worker_num, self.sess) for i in xrange(worker_num)]
        for i in xrange(worker_num):
            ps_t[i].start()
        
        #COST_PER_COMMIT = 78.0 # COST_PER_COMMIT is 78 M
        #if(self.band_width_limit == None):
        #    c_target_max_by_bandwidth = 100000.0
        #else: # specify the constrait of bandwidth in the form of x M/s
        #    c_target_max_by_bandwidth = float(sys.argv[1]) * self.check_period / COST_PER_COMMIT
        global isfirst, oktotest
        check_cnt = 0
        while(not self.trainingEndorNot):
            print('%%%%%%%%%%%%%%%%%%%%% main run %%%%%%%%%%%%%%%%%%%%')
            time.sleep(2)
            #print('is first',isfirst)
            if(not isfirst):
              if(oktotest):
                check_cnt = check_cnt + 1
                #time.sleep(200)
                #min_step = min(local_step_list)
                print('check_cnt',check_cnt)
                '''
                if(check_cnt % 1 == 0):
                    self.allStop(ps_t)
                    self.get_global_loss()
                    self.allStart(ps_t)
                    print("SSP:min_step %d\n" % (min_step))
                    
                    self.record_info()
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                '''
                if(check_cnt % 2 == 0):  # % 5s
                    ## for record
                    #self.record_info()
                    self.get_global_loss()
                    for i in xrange(worker_num):
                        ps_t[i].isCheckOrStop = 0
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            
        print("PS ends")
        for i in xrange(worker_num):
            ps_t[i].isCheckOrStop = -1
        for i in xrange(worker_num):
            ps_t[i].join()
        #self.f_wkloss_pushnum.close()
        self.f_time.close()
        #self.f_time2.close()
        self.f_global_loss.close()
        #self.f_global_eval.close()


