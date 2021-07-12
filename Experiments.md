# Experiments

### Time and accuracy

We did several experiments for measuring performance in terms of time and accuracy of:

* GPU on stat machine vs. GPU on the cluster

|                     | epoch | accuracy | loss   | time   |
|---------------------|-------|----------|--------|--------|
| GPU on stat machine | 50    | 65.1%    | 0.5931 | 57m8s  |
| GPU on cluster      | 97    | 68.8%    | 0.5805 | 34m41s |


* CPU on the cluster vs. GPU on the cluster

|                | epoch | accuracy | loss   | time   |
|----------------|-------|----------|--------|--------|
| CPU on cluster | 100   | 70.1%    | 0.5663 | 46m11s |
| GPU on cluster | 100   | 70.2%    | 0.5654 | 40m56s |


### Network usage

We investigated how the batch size affects the network usage for the Parameter Server (ps) when using ParameterServerStrategy as the distribution strategy.


### GPU usage

We looked at the GPU usages on the stat machine and on the cluster to see if there is any difference for the same training procedures. 
