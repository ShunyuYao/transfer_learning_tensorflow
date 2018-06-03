import tensorflow as tf
import tensornets as nets
import numpy as np
import pandas as pd

"""
fc_mid_layer is the layer before the last fc for softmax
this value means the hidden units of the layer
0 for no fc_mid_layers
"""

status = 'train'  # train for training, restore for restoring
input_size = 224
img_class = 100
channels = 3
batch_size = 10
droprate = 0
fc_mid_layers = 0
model_name = 'Densenet201'
num_train_vars = 150     # the last nums of trainable variables

test_df = pd.read_table('../datasets/test.txt',
                         sep=' ', header=None)
test_df.columns = ['filename']
test_name = test_df.copy()

def add_filepath(x):
    return '../datasets/test/' + x
test_df['filename'] = test_df['filename'].apply(add_filepath)
test_num = test_df.shape[0]

def _test_parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=channels)
  image_decoded = tf.cast(image_decoded, tf.float32)
  image_resized = tf.image.resize_images(image_decoded, [input_size, input_size])
  image_resized = tf.minimum(image_resized, 255.0)
  image_resized = tf.maximum(image_resized, 0.0)

  return image_resized

# A vector of filenames.
test_filename = tf.constant(test_df['filename'].values)

test_dataset = tf.data.Dataset.from_tensor_slices(test_filename)
test_dataset = test_dataset.map(_test_parse_function)
test_dataset = test_dataset.batch(batch_size)

iterator = tf.data.Iterator.from_structure(test_dataset.output_types,
                                           test_dataset.output_shapes)
next_element = iterator.get_next()

test_init_op = iterator.make_initializer(test_dataset)

# transfer learn
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, channels])

model = nets.DenseNet201(inputs, is_training=True, classes=img_class, stem=True)
top_layer = tf.reduce_mean(model, [1, 2], name='avgpool')
if droprate > 0:
    top_layer = tf.layers.dropout(top_layer, droprate)
if fc_mid_layers > 0:
    top_layer = tf.contrib.layers.fully_connected(top_layer, fc_mid_layers)
top_layer = tf.contrib.layers.fully_connected(top_layer, img_class)
top_layer = tf.contrib.layers.softmax(top_layer)

saver = tf.train.Saver()
test_label = pd.Series()
with tf.Session() as sess:
    saver.restore(sess, '../' + model_name + '/model.ckpt')
    print('model restore sucess')
    sess.run(test_init_op)
    print('starting to predict, please wait')
    while True:
        try:
            x_test = sess.run(next_element)
            x_test = model.preprocess(x_test)
            test_scores = sess.run(top_layer, {inputs: x_test})
            test_preds = np.argmax(test_scores, axis=1)
            test_preds += 1
            test_label = pd.concat([test_label, pd.Series(test_preds)])
        except tf.errors.OutOfRangeError:
            break

test_label.reset_index(inplace=True, drop=True)
pred_df = pd.concat([test_name, test_label], axis=1)
pred_df.to_csv('../predict/' + model_name + '.csv', header=None, index=False, sep=' ')
print('predict results saved')
