import tensorflow as tf
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.layers as layers

import cv2 as cv

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL = 224,224,3
tf.app.flags.DEFINE_string('tfrecords_path', r'~/data/cats_and_dogs/tfrecord', 'initial model dir')
tf.app.flags.DEFINE_string('tfrecords_file_name', 'train', 'training data dir')
# tf.app.flags.DEFINE_string('input_validation_data_path', 'data/test/inception_v4', 'validation data dir')
# tf.app.flags.DEFINE_string('output_model_path', r'model', 'output model dir')
# tf.app.flags.DEFINE_bool('save_model_every_epoch', True, 'whether save model every epoch')
# tf.app.flags.DEFINE_string('dict_path', 'l3g.txt', 'path of x_letter dict')
# tf.app.flags.DEFINE_string('log_dir', r'model', 'folder to save checkpoints')
tf.app.flags.DEFINE_integer('epochs', 1, 'epochs')
# tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')  #0.002
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
# tf.app.flags.DEFINE_integer('val_batch_size', 512, 'validation batch size')

FLAGS = tf.app.flags.FLAGS

def parse_example_train(serialized_example):
    features={'image':tf.FixedLenFeature([],tf.string),
         'label':tf.FixedLenFeature([],tf.int64)}
    example = tf.parse_single_example(serialized_example, features)
    return example

def distorted_input(filename, batch_size):
    """建立一个乱序的输入
    
    参数:
      filename: tfrecords文件的文件名. 注：该文件名仅为文件的名称，不包含路径和后缀
      batch_size: 每次读取的batch size
      
    返回:
      images: 一个4D的Tensor. size: [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
      labels: 1D的标签. size: [batch_size]
    """
    filename = os.path.join(FLAGS.tfrecords_path, (filename + '.tfrecords'))
    print(filename)
    ds = tf.data.TFRecordDataset(filename)
    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=8 * batch_size, count=FLAGS.epochs))
    ds = ds.apply(tf.data.experimental.map_and_batch(map_func=parse_example_train, batch_size=batch_size,
                                                   drop_remainder=False, num_parallel_calls=24))
    return ds

def inference(images, n_classes):
    # conv1, shape = [kernel_size, kernel_size, channels, kernel_numbers]
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")

    # pool1 && norm1
    with tf.variable_scope("pooling1_lrn") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")

    # pool2 && norm2
    with tf.variable_scope("pooling2_lrn") as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')

    # full-connect1
    with tf.variable_scope("fc1") as scope:
        reshape = layers.flatten(norm2)
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")

    # full_connect2
    with tf.variable_scope("fc2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name="fc2")

    # softmax
    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")
        # softmax_linear = tf.nn.softmax(softmax_linear)

    return softmax_linear


def losses(logits, labels):
    with tf.variable_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels)
        loss = tf.reduce_mean(cross_entropy)
    return loss


def evaluation(logits, labels):
    with tf.variable_scope("accuracy"):
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
    return accuracy

if __name__ == '__main__':
    dataset = distorted_input(FLAGS.tfrecords_file_name,batch_size=FLAGS.batch_size)
    iterator = dataset.make_initializable_iterator()
    
    batch_op = iterator.get_next()
    batch_input,label_train_batch = batch_op['label'],batch_op['label']
    images = tf.decode_raw(batch_input,tf.uint8)
    labels = tf.cast(label_train_batch,tf.int32)

    train_logits = inference(images,2)
    train_loss = losses(train_logits,label_train_batch)
    train_op = tf.train.AdamOptimizer(1e-4).minimize(train_loss)
    train_acc = evaluation(train_logits, labels)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(),iterator.initializer])
        while True:
            try:
                _,loss = sess.run([train_op, train_loss,train_acc])
            except tf.errors.OutOfRangeError:
                print("End of training.")
                break