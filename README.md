# ImagePipeline
An image pipeline for training image classifiers on the Hadoop cluster

## Getting started

Connect to stat1008 through ssh 

```bash
$ ssh stat1008.eqiad.wmnet
```

### Installation

First, clone the repository

```bash
$ git clone https://github.com/AikoChou/ImagePipeline.git
```

Setup and activate the virtual environment

```bash
$ cd ImagePipeline
$ conda-create-stacked venv
$ source conda-activate-stacked venv
```

Install the dependencies

```bash
$ export http_proxy=http://webproxy.eqiad.wmnet:8080
$ export https_proxy=http://webproxy.eqiad.wmnet:8080
$ pip install --ignore-installed -r requirements.txt
```

### Running the script

To run the script for training an image classifier, you need to

- Save your neural network model to JSON;
- Upload your data to HDFS;
- Setup the Config file.

First, Keras provides the ability to describe any model in JSON format through the `to_json()` function. This can be saved to a file and then loaded via the `model_from_json()` function, which will create a new model from the JSON specification.

```python
model = tf.keras.Sequential([
            ...
        ])

model_json = model.to_json()
with open('keras_model/model.json', 'w') as json_file:
    json_file.write(model_json)
```

Upload the prepared training and test data saved as TFRecord files to HDFS.

```bash
$ hadoop fs -copyFromLocal your_data.tfrecords folder_on_hdfs/
```

The Config file contains parameters for various purposes in the pipeline. The basic is to modify `_GLOBAL_CONFIG`, `_DATA_CONFIG`, and `_MODEL_CONFIG`.

**_GLOBAL_CONFIG**
- `name` the name of the yarn application
- `hdfs_dir` the location on HDFS at which the model and its checkpoints will be saved

```python
_GLOBAL_CONFIG = dict(
    name = 'ImagePipeline',
    hdfs_dir = f'{cluster_pack.get_default_fs()}user/{USER}/tf_yarn/tf_yarn_{int(datetime.now().timestamp())}'
)
```

**_DATA_CONFIG**

- `train_data` the location for the training data on HDFS. If there are multiple files, a list is given
- `eval_data` the location for the evaluation data on HDFS. If there are multiple files, a list is given
- `img_size` the shape for the image data
- `buffer_size` for shuffling

```python
_DATA_CONFIG = dict(
    train_data = [f'{cluster_pack.get_default_fs()}user/{USER}/pixels-160x160-shuffle-000.tfrecords'],
    eval_data = [f'{cluster_pack.get_default_fs()}user/{USER}/pixels-160x160-shuffle-001.tfrecords'],
    img_size = (160, 160, 3)
    buffer_size = 1000
)
```

**_MODEL_CONFIG**

- `weights_to_load` warm starts from pre-trained weights or the weights saved in a checkpoint on HDFS
- `load_var_name` whether to load a dict of name mapping for the name of the variable between previous checkpoint (or pre-trained model) and current model. If true, a 'var_name.json' file needs to provided
- `layer_to_train` specify a certain layer to train, freeze other layers. It is used in transfer learning/fine-tuning, to train the top layer only
- `train_steps` number of total steps for which to train model
- `eval_steps` number of steps for which to evaluate model. If None, evaluates the whole eval_data
- `batch_size` batch size to use. Note to adjust the batch size according to the device type (CPU or GPU), otherwise you may encounter network problems.[1]

```python
_MODEL_CONFIG = dict(
    weights_to_load = f'{cluster_pack.get_default_fs()}user/{USER}/mobilenet/variables/variables',
    load_var_name = True,
    layer_to_train = 'dense',
    train_steps = 1000,
    eval_steps = None,
    batch_size = 256,
    learning_rate = 1e-3,
    optimizer = 'gradient_descent',
    loss_fn = 'binary_crossentropy',
    metric_fn = 'binary_accuracy'
)
```

Other blocks of configuration can remain unchanged, such as `_RESOURCE_CONFIG` that provide default settings to resources for distributed tasks, and `_HADOOP_ENV_CONFIG` that set up all the environment variables to have Tensorflow working with HDFS.

After setting the above things, we can run the script. You can choose from two versions, one is running on CPU nodes, and the other is running on GPU nodes.

```bash
$ python scripts/train.py # for CPU nodes
$ python scripts/train_on_gpu.py # GPU nodes
```

### Checking out the performance

To know the accuracy of the model after training, we can look at the yarn logs.

```bash
$ yarn logs -applicationId application_xxxxxxxxxxxxx_xxxxx
```

In the **evaluator.log**, we can see the evaluation accuracy and loss for the final training step.

```bash
...
2021-06-30 14:41:30,411:INFO:tensorflow: Finished evaluation at 2021-06-30-14:41:30
2021-06-30 14:41:30,412:INFO:tensorflow: Saving dict for global step 1007: binary_accuracy = 0.6373899, global_step = 1007, loss = 0.63418937
2021-06-30 14:41:30,474:INFO:tensorflow: Saving 'checkpoint_path' summary for global step 1007: hdfs://analytics-hadoop/user/aikochou/tf_yarn/tf_yarn_1625063221/model.ckpt-1007
2021-06-30 14:41:30,479:DEBUG:tensorflow: Calling exporter with the `is_the_final_export=True`.
2021-06-30 14:41:30,480:INFO:tensorflow: Waiting 453.976144 secs before starting next eval run.
2021-06-30 14:49:04,488:INFO:tensorflow: Exiting evaluation, global_step=1007 >= train max_steps=1000
2021-06-30 14:49:04,498:INFO:__main__: evaluator:0 SUCCEEDED
```
In addition, the locations at which the checkpoints saved are shown. By setting the `weights_to_load` in the config file to the latest checkpoint,  you can warm start from the model status and continue to train the model.


In the **chief.log**, we can see the initial loss and the loss for final step.

```bash
2021-06-30 14:27:25,565:INFO:tensorflow: loss = 0.87347865, step = 0
2021-06-30 14:28:11,977:INFO:tensorflow: global_step/sec: 2.30507
2021-06-30 14:28:49,234:INFO:tensorflow: global_step/sec: 2.81827
2021-06-30 14:29:30,823:INFO:tensorflow: global_step/sec: 2.83734
2021-06-30 14:30:23,872:INFO:tensorflow: global_step/sec: 2.58249
2021-06-30 14:31:04,625:INFO:tensorflow: global_step/sec: 2.82185
2021-06-30 14:31:39,712:INFO:tensorflow: global_step/sec: 3.10662
2021-06-30 14:31:41,813:INFO:tensorflow: loss = 0.64253294, step = 694 (256.248 sec)
2021-06-30 14:32:15,668:INFO:tensorflow: global_step/sec: 3.05927
2021-06-30 14:32:19,758:INFO:tensorflow: Calling checkpoint listeners before saving checkpoint 817...
2021-06-30 14:32:19,758:INFO:tensorflow: Saving checkpoints for 817 into hdfs://analytics-hadoop/user/aikochou/tf_yarn/tf_yarn_1625063221/model.ckpt.
2021-06-30 14:32:20,674:INFO:tensorflow: Calling checkpoint listeners after saving checkpoint 817...
2021-06-30 14:32:50,571:INFO:tensorflow: global_step/sec: 3.06574
2021-06-30 14:33:23,267:INFO:tensorflow: Calling checkpoint listeners before saving checkpoint 1007...
2021-06-30 14:33:23,268:INFO:tensorflow: Saving checkpoints for 1007 into hdfs://analytics-hadoop/user/aikochou/tf_yarn/tf_yarn_1625063221/model.ckpt.
2021-06-30 14:33:24,361:INFO:tensorflow: Calling checkpoint listeners after saving checkpoint 1007...
2021-06-30 14:33:24,471:INFO:tensorflow: Loss for final step: 0.6461505.
2021-06-30 14:33:24,482:INFO:__main__: chief:0 SUCCEEDED
```
