import config
import tensorflow as tf
import os
import json

cfg = config.get_config()

def _binary_crossentropy(labels, logits):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, logits)

def _sparse_categorical_crossentropy(labels, logits):
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, logits)

def _binary_accuracy(labels, logits):
    return tf.compat.v1.metrics.accuracy(labels=tf.cast(labels, tf.int32),
                                         predictions=tf.cast(tf.math.greater(tf.nn.sigmoid(logits),
                                                                             tf.constant([0.5])), tf.int32))
def _categorical_accuracy(labels, logits):
    return tf.compat.v1.metrics.accuracy(labels=tf.cast(labels, tf.int32),
                                         predictions=tf.argmax(logits, axis=1))

def _gradient_descent_optimizer(learning_rate):
    return tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)

optimizers = {'gradient_descent': _gradient_descent_optimizer}

losses = {'binary_crossentropy': _binary_crossentropy,
          'sparse_categorical_crossentropy': _sparse_categorical_crossentropy}

metrics = {'binary_accuracy': _binary_accuracy,
           'categorical_accuracy': _categorical_accuracy}

def model_fn(features, labels, mode):
    model_file = os.path.join('keras_model', 'model.json') if os.path.isdir('keras_model') else 'model.json'
    with open(model_file, 'r') as f:
        model_json = f.read()
    model = tf.keras.models.model_from_json(model_json)
    logits = model(features)

    optimizer = optimizers[cfg.opt](learning_rate=cfg.learning_rate)
    loss = losses[cfg.loss_fn](labels, logits)
    eval_metric = {cfg.metric_fn: metrics[cfg.metric_fn](labels, logits)}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric)

    if cfg.layer_to_train:
        training_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                    cfg.layer_to_train)
        train_op = optimizer.minimize(
                  loss, tf.compat.v1.train.get_or_create_global_step(), var_list=training_vars)
    else:
        train_op = optimizer.minimize(
                  loss, tf.compat.v1.train.get_or_create_global_step())

    return tf.estimator.EstimatorSpec(
              mode=mode,
              loss=loss,
              train_op=train_op
            )
