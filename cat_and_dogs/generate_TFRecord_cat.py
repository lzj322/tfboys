import tensorflow as tf
import os
import glob
import numpy as np
from param import FLAGS
import cv2 as cv


rawdata_path = FLAGS.input_training_data_path
tfrecords_path = FLAGS.output_model_path
tf_name = FLAGS.tf_name

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL = 224,224,3


def read_images(path):
    '''Read image from path'''
    filenames = glob.glob(os.path.join(path,'*.jpg'))
    num_file = len(filenames)

    images = np.zeros((num_file,IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL),dtype=np.uint8)
    labels = np.zeros((num_file,),dtype=np.uint8)

    for index, filename in enumerate(filenames):
        img = cv.imread(os.path.join(path,filename))
        try:
            img = cv.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
        except:
            continue
        images[index] = img

        if filename[:3] == 'cat':
            labels[index] = int(0)
        else:
            labels[index] = int(1)

        if index%1000 == 0:
            print("Reading the %d th image" % index)

    return images, labels, num_file

def convert(images,labels,num_examples, destination):
    ''' convert images into tfrecords'''
    filename = destination
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_examples):
        image = images[index].tostring()
        label = labels[index]
        feature_dict = {
            'image':tf.train.Feature(bytes_list= tf.train.BytesList(value=[image])),
            'label':tf.train.Feature(int64_list= tf.train.Int64List(value=[label]))
        }
        example = tf.train.Example(features=tf.train.Features(feature = feature_dict))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    print('running')
    tf.gfile.MakeDirs(tfrecords_path)
    print('Transfer images to TFRecords')
    images,labels,num_file = read_images(rawdata_path)
    convert(images,labels,num_file,os.path.join(tfrecords_path,tf_name))
    print('End transfering')