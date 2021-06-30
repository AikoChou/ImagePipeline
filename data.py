import config
import tensorflow as tf

cfg = config.get_config()

def parse_example(example):
    features = tf.io.parse_single_example(example, features={
        "image": tf.io.FixedLenFeature([cfg.img_size * cfg.img_size * 3], dtype=tf.float32),
        "label": tf.io.FixedLenFeature([], dtype=tf.float32)
    })
    x = tf.reshape(features["image"], [cfg.img_size, cfg.img_size, 3])
    y = features["label"]
    return x, y

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

def input_fn(mode, input_context=None):
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = tf.data.TFRecordDataset(cfg.train_data).map(parse_example)
        if input_context:
            dataset = dataset.shard(input_context.num_input_pipelines,
                                    input_context.input_pipeline_id)
        return dataset.map(scale).shuffle(cfg.buffer_size).repeat(None).batch(cfg.batch_size).prefetch(1)

    else:
        dataset = tf.data.TFRecordDataset(cfg.eval_data).map(parse_example)
        if input_context:
            dataset = dataset.shard(input_context.num_input_pipelines,
                                    input_context.input_pipeline_id)
        return dataset.map(scale).shuffle(cfg.buffer_size).batch(cfg.batch_size).prefetch(1)