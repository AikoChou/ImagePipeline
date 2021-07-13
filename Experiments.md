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


### Network usage

We investigated how the batch size affects the network usage for the Parameter Server (ps) when using ParameterServerStrategy as the distribution strategy.


### GPU usage

We looked at the GPU usages on the stat machine and on the cluster to see if there is any difference for the same training procedures. 



