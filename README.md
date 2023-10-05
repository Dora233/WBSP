# WBSP
Official implementation of "WBSP: Worker-Busy Synchronous Parallel Mechanism for Distributed Machine Learning".<br>
>The parameter server is widely used in distributed machine learning to speed up training. However, the increasing heterogeneity of workers' computing capabilities makes parameter synchronization challenging. To addressing the issue of stragglers, we propose the worker-busy synchronous parallel (WBSP) mechanism based on asymmetric upload and pull-down. It adjusts the communication pace of fast workers by decoupling the gradient upload and model pull-down into independent and asymmetric parts. This eliminates the waiting time of fast workers during the synchronization process, allowing them to complete multiple rounds of local training and upload multiple gradients to the server within the original waiting time, greatly improving the computational efficiency. At the same time, the global update on the server is triggered based on the event of uploading gradients by the slowest workers, avoiding the introduction of stale gradients and ensuring the consistency of local and global model updates for all workers. Based on WBSP, an optimized version of the mechanism is proposed to address the issue of limited acceleration caused by stragglers. The optimization focuses on parallelizing communication and computation to shorten the global synchronization interval, improving the training speed and synchronization efficiency. We perform a theoretical analysis of the convergence of WBSP and provide a comparison between existing mechanisms. Extensive experiments on classical models and datasets demonstrate the effectiveness of WBSP for various distributed training scenarios and verify that WBSP can provide 2.07$\times$ speedup compared to the fastest method.<br>

We use Intel Xeon CPU containing a clock rate of 3.0 GHz with 32 cores and utilize 8 Nvidia Tesla V100 GPUs to accelerate training.
The OS system is Ubuntu18.04. The driver version is 440.118.02 and CUDA version is 10.2.
For the base settings, the number of the illustrative workers in each round of training is set to ten.<br>

## Run Simulation
WBSP for increasing heterogeneity on CIFAR-10 can be tested by runing the following commands to submit the task:
```
CUDA_VISIBLE_DEVICES= nohup python -u runcifar10gpu.py  --job_name=ps --worker_num=2 --base_dir=wbsp/ --port_base=2606 >ps.log 2>&1 & 
CUDA_VISIBLE_DEVICES=0 nohup python -u runcifar10gpu.py  --job_name=worker --worker_index=0 --base_dir=wbsp/ --port_base=2606 >wk0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u runcifar10gpu.py  --job_name=worker --worker_index=1 --base_dir=wbsp/ --port_base=2606 >wk1.log 2>&1 &
```
Similarly, "runimagenetgpu.py" is for training on the ImageNet dataset. <br>

## Acknowledgements
Thanks to Hanpeng Hu, Dan Wang, Chuan Wu for their AAAI'20 paper [Distributed Machine Learning through Heterogeneous Edge Systems](https://aaai.org/papers/07179-distributed-machine-learning-through-heterogeneous-edge-systems/). <br>
