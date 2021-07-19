# Experiments

### Time and accuracy

We did several experiments for measuring performance in terms of time and accuracy of:

**GPU on stat machine vs. GPU on the cluster**

|                     | epoch | accuracy | loss   | time   |
|---------------------|-------|----------|--------|--------|
| GPU on stat machine | 50    | 65.1%    | 0.5931 | 57m8s  |
| GPU on cluster      | 97    | 68.8%    | 0.5805 | 34m41s |

In this experiment, there is 1 GPU on the stat machine, and the intra- and inter- parallelism threads for tensorflow were set to 6. The image quality classifier was training with Keras API with model.fit. On the other hand, on the cluster, the classifier was training with Estimators API with the Parameter Server Strategy on 6 GPUs.

The batch size for both was set to 512, on the stat the classifier was trained for 50 epochs, and on the cluster, the classifier was trained for around 100 epochs. The training dataset and the evaluation dataset have about 20k images each, and the label distribution is balanced. 

The result shows the classifier training on the cluster can achieve more or less the same accuracy, and better accuracy with more epochs. Regarding elapsed time, the distributed training on the cluster is 3x faster than training on the stat machine.

**CPU on the cluster vs. GPU on the cluster**

|                | epoch | accuracy | loss   | time   |
|----------------|-------|----------|--------|--------|
| CPU on cluster | 100   | 70.1%    | 0.5663 | 46m11s |
| GPU on cluster | 100   | 70.2%    | 0.5654 | 40m56s |

In this experiment, one image quality classifier was training using CPU nodes, while another was using GPU nodes. In order to allow the Hadoop cluster to use the GPUs available on some workers, we use Yarn node labels where CPU-only nodes are unlabelled, while the GPU ones have a label. Furthermore, GPU nodes are bound to a separate queue called "fifo" which is different from the default one. Refer to [Phab task](https://phabricator.wikimedia.org/T276791)

Both classifiers were training with Estimators API with the Parameter Server Strategy. For the GPU setting, due to only having 6 GPU nodes, we chose to run the compute-heavy "chief"(1), "worker"(4), and "evaluator"(1) tasks on GPU, while keeping "ps"(2) on CPU. The batch size for both was set to 256 and the number of epoch for training was set to 100. We used the same training dataset and the evaluation dataset which are the same as the former experiment.

The result shows classifiers can achieve the same accuracy and loss either on the CPU-only nodes or GPU nodes. Regarding the elapsed time, it is surprising that training on GPU nodes doesn't show more advantages in terms of efficiency which is only 5 minutes faster than training on CPU-only nodes.


### Issue: Parameter Server being the bottleneck for computation

One issue we encountered in the experiment was the surge in network traffic ~1GB/s towards GPU nodes. The reason is that the worker dispatching weights to the Parameter Server was overloaded and saturated the network. Therefore, we looked for ways to reduce or re-distribute this load. 

First, we tried to move the Parameter Server to CPU nodes and increase their number. However, we found that although non-GPU nodes can be assigned to non-GPU tasks, the load cannot be shared evenly among multiple parameter servers.

Next, we tried to increase the batch size used for training because we believe that a small batch size results in frequent updates of model variables. Under the ParameterServerStrategy, workers communicate with PS to save and get model variables, so frequent updates cause high network traffic. Finally, we observed a significant decrease in network traffic by increasing the batch size from 32 to 512.

|PS 1               |PS 2  | 
|---------------------|-------|
| <img width="500" alt="gpu_ps0" src="https://user-images.githubusercontent.com/14852065/126120130-be282bc0-69c6-4578-9413-15bcb935c048.png"> | <img width="500" alt="gpu_ps1" src="https://user-images.githubusercontent.com/14852065/126120148-4db17d1d-6b15-416b-bd75-8f92b5c2ef6e.png">    |

### GPU usage

We looked at the GPU usages on the stat machine vs on the cluster. We observed that on the stat machine, the GPU usage is not sustained, instead, there are peaks one after another. These peaks sometimes appear frequently but sometimes occasionally. The following figures are two tasks training under the same parameters at different times:

|test 1               |test 2  | 
|---------------------|-------|
| <img width="500" alt="local_batch_512" src="https://user-images.githubusercontent.com/14852065/126119017-8bf04c01-9913-447c-8bae-fbfcd2328f76.png"> | <img width="500" alt="local_gpu_usage" src="https://user-images.githubusercontent.com/14852065/126119176-af51a4d5-f7ac-411c-8e1e-47acdca5a8aa.png">    |

On the cluster, the GPU usage on different nodes shows similar patterns with the stat machine -- occasional peaks:

|chief               |worker 1 |worker 2 | 
|---------------------|-------|-------|
| <img width="300" alt="gpu_chief" src="https://user-images.githubusercontent.com/14852065/126119984-34e777c7-9dd8-4dc8-96e5-2503741b7d59.png"> | <img width="300" alt="gpu_worker0" src="https://user-images.githubusercontent.com/14852065/126120042-f02cb1b3-93c5-40ae-b282-97d5e331f83a.png"> | <img width="300" alt="gpu_worker1" src="https://user-images.githubusercontent.com/14852065/126120012-498e3dc5-9b6b-43aa-a7ad-5cd38447410e.png"> |

|evaluator               |worker 3 | worker 4 | 
|---------------------|-------|-------|
| <img width="300" alt="gpu_evaluator" src="https://user-images.githubusercontent.com/14852065/126119989-72cbd437-2628-4404-94f9-1cfc263534d9.png"> | <img width="300" alt="gpu_worker2" src="https://user-images.githubusercontent.com/14852065/126120030-ffec81be-4ec6-4591-b26a-61cf71de73f0.png"> | <img width="300" alt="gpu_worker3" src="https://user-images.githubusercontent.com/14852065/126114802-b6f47fb3-9758-47cb-bf77-2cf24d95327d.png"> |
