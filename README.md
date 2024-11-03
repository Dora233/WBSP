# WBSP: Addressing stragglers in distributed machine learning with worker-busy synchronous parallel
Official implementation of WBSP published in Parallel Computing. [[Paper]]([https://authors.elsevier.com/a/1jONVcA5lHSOr](https://www.sciencedirect.com/science/article/abs/pii/S0167819124000309))<br>
>Parameter server is widely used in distributed machine learning to accelerate training. However, the increasing heterogeneity of workers' computing capabilities leads to the issue of stragglers, making parameter synchronization challenging. To address this issue, we propose a solution called Worker-Busy Synchronous Parallel (WBSP). This approach eliminates the waiting time of fast workers during the synchronization process and decouples the gradient upload and model download of fast workers into asymmetric parts. By doing so, it allows fast workers to complete multiple steps of local training and upload more gradients to the server, improving computational resource utilization. Additionally, the global model is only updated when the slowest worker uploads the gradients, ensuring the consistency of global models that are pulled down by all workers and the convergence of the global model. Building upon WBSP, we propose an optimized version to further reduce the communication overhead. It enables parallel execution of communication and computation tasks on workers to shorten the global synchronization interval, thereby improving training speed. We conduct theoretical analyses for the proposed mechanisms. Extensive experiments verify that our mechanism can reduce the required time to achieve the target accuracy by up to 60\% compared with the fastest method and increase the proportion of computation time from 55\%--72\% in existing methods to 91\%.<br>

We use Intel Xeon CPU containing a clock rate of 3.0 GHz with 32 cores and utilize Nvidia Tesla V100 GPUs to accelerate training.
The OS system is Ubuntu18.04. The driver version is 440.118.02 and CUDA version is 10.2.<br>

## Run Simulation
WBSP for increasing heterogeneity on CIFAR-10 can be tested by runing the following commands to submit the task:
```
CUDA_VISIBLE_DEVICES= nohup python -u runcifar10gpu.py  --job_name=ps --worker_num=2 --base_dir=wbsp/ --port_base=2606 >ps.log 2>&1 & 
CUDA_VISIBLE_DEVICES=0 nohup python -u runcifar10gpu.py  --job_name=worker --worker_index=0 --base_dir=wbsp/ --port_base=2606 >wk0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u runcifar10gpu.py  --job_name=worker --worker_index=1 --base_dir=wbsp/ --port_base=2606 >wk1.log 2>&1 &
```

## Citation
If you find WBSP useful for your research and applications, please cite using this BibTeX:
```
@article{YANG2024103092,
author = {Duo Yang and Bing Hu and An Liu and A-Long Jin and Kwan L. Yeung and Yang You},
title = {WBSP: Addressing stragglers in distributed machine learning with worker-busy synchronous parallel},
journal = {Parallel Computing},
volume = {121},
pages = {103092},
year = {2024},
issn = {0167-8191},
doi = {https://doi.org/10.1016/j.parco.2024.103092},
url = {https://www.sciencedirect.com/science/article/pii/S0167819124000309}
}
```

## Acknowledgements
We appreciate the help from Hanpeng Hu, Dan Wang, Chuan Wu for their AAAI'20 paper [Distributed Machine Learning through Heterogeneous Edge Systems](https://aaai.org/papers/07179-distributed-machine-learning-through-heterogeneous-edge-systems/). <br>
