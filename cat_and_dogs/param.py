import tensorflow as tf

tf.app.flags.DEFINE_string('input_previous_model_path', 'model/', 'initial model dir')
tf.app.flags.DEFINE_string('input_previous_model_name', 'model_final', 'initial model name')
tf.app.flags.DEFINE_string('log_dir', 'summary/', 'initial summary dir')

tf.app.flags.DEFINE_string('input_training_data_path', 'data/train', 'training data dir')
tf.app.flags.DEFINE_string('input_validation_data_path', 'data/test', 'validation data dir')
tf.app.flags.DEFINE_string('output_model_path', 'model/', 'output model dir')
tf.app.flags.DEFINE_string('tf_name', 'train.tfrecords', 'output filename')

tf.app.flags.DEFINE_integer('file_idx_1', 0, 'num of score files')
tf.app.flags.DEFINE_integer('file_idx_2', 128, 'num of score files')
tf.app.flags.DEFINE_integer('file_offset', 0, 'num of score files')

FLAGS = tf.app.flags.FLAGS


