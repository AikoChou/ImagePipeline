import os
import types
import getpass
from functools import reduce
from datetime import datetime
import cluster_pack

USER = getpass.getuser()

_GLOBAL_CONFIG = dict(
    name = 'ImagePipeline',
    hdfs_dir = f'{cluster_pack.get_default_fs()}user/{USER}/tf_yarn/tf_yarn_{int(datetime.now().timestamp())}'
)

_DATA_CONFIG = dict(
    train_data = [f'{cluster_pack.get_default_fs()}user/{USER}/pixels-160x160-shuffle-000.tfrecords'],
    eval_data = [f'{cluster_pack.get_default_fs()}user/{USER}/pixels-160x160-shuffle-001.tfrecords'],
    img_size = 160,
    buffer_size = 1000
)

_MODEL_CONFIG = dict(
    weights_to_load = f'{cluster_pack.get_default_fs()}user/{USER}/mobilenet/variables/variables',
    load_var_name = True,
    layer_to_train = 'dense',
    opt = 'gradient_descent',
    loss_fn = 'binary_crossentropy',
    metric_fn = 'binary_accuracy',
    batch_size = 256,
    learning_rate = 1e-3,
    train_steps = 1000,
    eval_steps = None
)

_PYENV_CONFIG = dict(
    gpu_env = '/home/aikochou/tf-yarn-rocm.zip',
    cpu_env = '/home/aikochou/tf-yarn-env.zip'
)

_RESOURCE_CONFIG = dict(
    gpu = dict(        
        queue = 'fifo',
        chief = dict(memory='8 GiB', vcores=4, instances=1),
        worker = dict(memory='8 GiB', vcores=4, instances=4),
        ps = dict(memory='2 GiB', vcores=4, instances=2),
        evaluator = dict(memory='8 GiB', vcores=1, instances=1)
    ),
    cpu = dict(
        queue = 'default',
        chief = dict(memory='4 GiB', vcores=4, instances=1),
        worker = dict(memory='4 GiB', vcores=4, instances=4),
        ps = dict(memory='2 GiB', vcores=4, instances=2),
        evaluator = dict(memory='4 GiB', vcores=1, instances=1)
    )
)

_HADOOP_ENV_CONFIG = dict(
    java_home = '/usr/lib/jvm/java-8-openjdk-amd64',
    hadoop_home = '/usr/lib/hadoop',
    hadoop_conf_dir = '/usr/lib/hadoop/etc/hadoop',
    ld_library_path = ['${JAVA_HOME}/jre/lib/amd64/server', 
                       '/usr/lib'],
    classpath = '$(${HADOOP_HOME}/bin/hadoop classpath --glob)',
    krb5ccname = '/tmp/krb5cc_$(id -u)',
)

_MIOPEN_CONFIG = dict(
    miopen_user_db_path = f'/home/{USER}/.config/miopen',
    mipoen_disable_cache = '1'
)

Config = types.SimpleNamespace

def _inherit(base, child):
    ret = dict(base)  # shallow copy
    for k, v in child.items():
        if k in ret:
            if isinstance(v, list):
                v = ret[k] + v
            elif isinstance(v, dict):
                v = dict(ret[k], **v)
        ret[k] = v
    return ret


def get_config(device='cpu'):
    resource_config = _RESOURCE_CONFIG[device]
    cfg = Config(device = device, **reduce(
        _inherit, [_GLOBAL_CONFIG, _HADOOP_ENV_CONFIG, _MIOPEN_CONFIG, 
                   _DATA_CONFIG, _MODEL_CONFIG, _PYENV_CONFIG, 
                   resource_config]))
    return cfg
