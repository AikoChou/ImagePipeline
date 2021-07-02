import os
import sys
import getpass
import json

_upper_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
if _upper_dir not in sys.path:
    sys.path.append(_upper_dir)

import config
import utils
import data
import model

from data import input_fn
from model import model_fn
from utils import NodeLabel
from tf_yarn import event, TaskSpec, Experiment, run_on_yarn, get_safe_experiment_fn
from tensorflow.estimator import RunConfig, WarmStartSettings
from tensorflow.estimator import Estimator, TrainSpec, EvalSpec
import tensorflow as tf

USER = getpass.getuser()
cfg = config.get_config('cpu')

os.environ['JAVA_HOME'] = cfg.java_home
os.environ['HADOOP_HOME'] = cfg.hadoop_home
os.environ['HADOOP_CONF_DIR'] = cfg.hadoop_conf_dir
for path in cfg.ld_library_path:
    utils.set_env_variable('LD_LIBRARY_PATH', '${LD_LIBRARY_PATH}:' + path)
utils.set_env_variable('CLASSPATH', cfg.classpath)
utils.set_env_variable('KRB5CCNAME', cfg.krb5ccname)

def experiment_fn() -> Experiment:
    run_config = RunConfig(model_dir=cfg.hdfs_dir, save_checkpoints_secs=300)

    if not cfg.weights_to_load: # cold start
        estimator = Estimator(model_fn=model_fn, config=run_config)

    elif not cfg.load_var_name: # warm-starts all variables in the trainable variables
        ws = WarmStartSettings(ckpt_to_initialize_from=cfg.weights_to_load,
                               vars_to_warm_start='.*',
                               var_name_to_prev_var_name=None)
        estimator = Estimator(model_fn=model_fn, config=run_config, warm_start_from=ws)

    else: # warm-starts the variables specified
        var_file = os.path.join('keras_model', 'var_name.json'
                               ) if os.path.isdir('keras_model') else 'var_name.json'
        with open(var_file) as f:
            var_name = json.load(f)
        ws = WarmStartSettings(ckpt_to_initialize_from=cfg.weights_to_load,
                               vars_to_warm_start=list(var_name.keys()),
                               var_name_to_prev_var_name=var_name)
        estimator = Estimator(model_fn=model_fn, config=run_config, warm_start_from=ws)

    experiment = Experiment(estimator,
                            TrainSpec(input_fn,
                                      max_steps=cfg.train_steps),
                            EvalSpec(input_fn,
                                     steps=cfg.eval_steps))
    return experiment

def get_safe_exp_fn():
    return get_safe_experiment_fn("train.experiment_fn")

def main(device):
    experiment_fn()
    pyenv_zip_path = {NodeLabel.GPU: cfg.gpu_env, NodeLabel.CPU: cfg.cpu_env}
    if device == 'gpu':
        task_specs = {
            'chief': TaskSpec(memory = cfg.chief['memory'],
                              vcores = cfg.chief['vcores'],
                              instances = cfg.chief['instances'],
                              label = NodeLabel.GPU),
            'worker': TaskSpec(memory = cfg.worker['memory'],
                               vcores = cfg.worker['vcores'],
                               instances = cfg.worker['instances'],
                               label = NodeLabel.GPU),
            'ps': TaskSpec(memory = cfg.ps['memory'],
                           vcores = cfg.ps['vcores'],
                           instances = cfg.ps['instances'],
                           label = NodeLabel.CPU),
            'evaluator': TaskSpec(memory = cfg.evaluator['memory'],
                                  vcores = cfg.evaluator['vcores'],
                                  instances = cfg.evaluator['instances'],
                                  label = NodeLabel.GPU)
        }
    else:
        task_specs = {
            'chief': TaskSpec(memory = cfg.chief['memory'],
                              vcores = cfg.chief['vcores'],
                              instances = cfg.chief['instances'],
                              label = NodeLabel.CPU),
            'worker': TaskSpec(memory = cfg.worker['memory'],
                               vcores = cfg.worker['vcores'],
                               instances = cfg.worker['instances'],
                               label = NodeLabel.CPU),
            'ps': TaskSpec(memory = cfg.ps['memory'],
                           vcores = cfg.ps['vcores'],
                           instances = cfg.ps['instances'],
                           label = NodeLabel.CPU),
            'evaluator': TaskSpec(memory = cfg.evaluator['memory'],
                                  vcores = cfg.evaluator['vcores'],
                                  instances = cfg.evaluator['instances'],
                                  label = NodeLabel.CPU)
        }
    upload_files = {
        os.path.basename(__file__): __file__,
        os.path.basename(config.__file__): config.__file__,
        os.path.basename(model.__file__): model.__file__,
        os.path.basename(data.__file__): data.__file__,
        os.path.basename(utils.__file__): utils.__file__,
        'model.json': f'/srv/home/{USER}/ImagePipeline/keras_model/model.json'}
    if cfg.load_var_name:
        upload_files['var_name.json'] = f'/srv/home/{USER}/ImagePipeline/keras_model/var_name.json'

    run_on_yarn(
        pyenv_zip_path,
        get_safe_exp_fn(),
        task_specs = task_specs,
        queue = cfg.queue,
        name = cfg.name,
        files = upload_files,
        pre_script_hook = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:' +
                          ':'.join(cfg.ld_library_path) +
                          ' && export CLASSPATH=$CLASSPATH:`hadoop classpath --glob`'
    )


if __name__ == '__main__':
    main('cpu')
