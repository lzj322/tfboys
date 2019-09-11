import tensorflow as tf
import os
import glob
import numpy as np
# import matplotlib.pyplot as plt

import cv2 as cv

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL = 224,224,3
tf.app.flags.DEFINE_string('tfrecords_path', r'~/data/cats_and_dogs/tfrecord', 'initial model dir')
tf.app.flags.DEFINE_string('tfrecords_file_name', 'train', 'training data dir')
# tf.app.flags.DEFINE_string('input_validation_data_path', 'data/test/inception_v4', 'validation data dir')
# tf.app.flags.DEFINE_string('output_model_path', r'model', 'output model dir')
# tf.app.flags.DEFINE_bool('save_model_every_epoch', True, 'whether save model every epoch')
# tf.app.flags.DEFINE_string('dict_path', 'l3g.txt', 'path of x_letter dict')
# tf.app.flags.DEFINE_string('log_dir', r'model', 'folder to save checkpoints')
# tf.app.flags.DEFINE_integer('epochs', 1, 'epochs')
# tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')  #0.002
# tf.app.flags.DEFINE_integer('batch_size', 512, 'batch size')
# tf.app.flags.DEFINE_integer('val_batch_size', 512, 'validation batch size')

FLAGS = tf.app.flags.FLAGS
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL = 224,224,3

def parse_example_train(serialized_example):
    features={'image':tf.FixedLenFeature([],tf.string),
         'label':tf.FixedLenFeature([],tf.int64)}
    example = tf.parse_single_example(serialized_example, features)
    image = tf.decode_raw(example['image'],tf.uint8)
    image = tf.reshape(image, [IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNEL])
    image = tf.cast(image, tf.float32)
    label = tf.cast(example['label'],tf.int64)
    example = {'image':image,'label':label}
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
    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=8 * batch_size, count=2))
    ds = ds.apply(tf.data.experimental.map_and_batch(map_func=parse_example_train, batch_size=batch_size,
                                                   drop_remainder=False, num_parallel_calls=24))
    # ds = ds.map(lambda x:parse_example_train(x)).batch(2)
    return ds

if __name__ == '__main__':
    dataset = distorted_input(FLAGS.tfrecords_file_name,batch_size=4)
    iterator = dataset.make_one_shot_iterator()
    
    batch_input = iterator.get_next()
    with tf.Session() as sess:
        # sess.run([iterator.initializer])
        i = 0
        while i<4:
            example =  sess.run(batch_input)
            image, label = example['image'], example['label']
            print (image.shape,label)
            i+=1