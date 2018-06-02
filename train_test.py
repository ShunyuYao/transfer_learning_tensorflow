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
batch_size = 128
num_epochs = 500
fc_mid_layers = 0
verbose_iters = 100
model_name = 'Densenet201'
droprate = 0             # use dropout before fc, 0 for no dropout
buffer_size = 100        # the buffer size for shuffle
best_valid_acc = 0
learning_rate = 1e-4
random_seed = 600        # the random seed for selecting valid image
num_train_vars = 150     # the last nums of trainable variables
random_aug_thresh = 0.25 # the ratio of pics for random augmentation

train_df = pd.read_table('../../datasets/train.txt',
                         sep=' ', header=None)
train_df.columns = ['filename', 'label']
train_df['label'] = train_df['label'].apply(lambda x: x-1)

# 随机生成验证集
np.random.seed(random_seed)
valid_name = pd.DataFrame()
for _, df in train_df.groupby('label'):
    num = df.shape[0]
    select_num = np.random.randint(2, 5)
    index = np.random.choice(num, np.random.randint(2,4), replace=False)
    valid_name = pd.concat([valid_name, df.iloc[index, :]])
train_name = train_df[~(train_df['filename'].isin(valid_name['filename']))]

def add_filepath(x):
    return '../../datasets/train/' + x
train_name['filename'] = train_name['filename'].apply(add_filepath)
valid_name['filename'] = valid_name['filename'].apply(add_filepath)
train_num = train_name.shape[0]
valid_num = valid_name.shape[0]

def accuracy(labels, logits):
    with tf.name_scope('accuracuy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(labels, 1), tf.arg_max(logits, 1)), "float"))
    return accuracy

def loss(labels, logits):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return loss

summary_train_dir = '../' + model_name + '/train_logs'
summary_valid_dir = '../' + model_name + '/valid_logs'

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
# 生成data loader 含数据预处理（增强）
def _train_parse_function(filename, label):
  image_string = tf.read_file(filename)
  img = tf.image.decode_jpeg(image_string, channels=channels)
  img = tf.cast(img, tf.float32)
  img = tf.image.resize_images(img, [input_size, input_size])
  if random_aug_thresh > 0:
      thresh = np.random.rand()
      if thresh < random_aug_thresh:
          img = tf.image.random_brightness(img, 50)
          img = tf.image.random_contrast(img, 0.6, 1.4)
          img = tf.image.random_hue(img, 0.05)
  img = tf.minimum(img, 255.0)
  img = tf.maximum(img, 0.0)

  label_oneHot = tf.one_hot(label, depth=img_class, dtype=tf.uint8)
  return img, label_oneHot

def _valid_parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=channels)
  image_decoded = tf.cast(image_decoded, tf.float32)
  image_resized = tf.image.resize_images(image_decoded, [input_size, input_size])
  image_resized = tf.minimum(image_resized, 255.0)
  image_resized = tf.maximum(image_resized, 0.0)

  label_oneHot = tf.one_hot(label, depth=img_class, dtype=tf.uint8)
  return image_resized, label_oneHot

def _random_shear(img, label):
    image = tf.keras.preprocessing.image.random_shear(
        img,
        50,
        row_axis=1,
        col_axis=0,
        channel_axis=2,
        fill_mode='nearest'
    )
    return image,label
# A vector of filenames.
train_filenames = tf.constant(train_name['filename'])
valid_filenames = tf.constant(valid_name['filename'])
# `labels[i]` is the label for the image in `filenames[i].
train_labels = tf.constant(train_name['label'])
valid_labels = tf.constant(valid_name['label'])

train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames,train_labels))
train_dataset = train_dataset.map(_train_parse_function)
# train_dataset = train_dataset.map(
#     lambda filename, label: tuple(
#         tf.py_func(_random_shear, [filename, label], [tf.float32, tf.uint8])
#     ))
train_dataset = train_dataset.repeat().batch(batch_size).shuffle(buffer_size=buffer_size)

valid_dataset = tf.data.Dataset.from_tensor_slices((valid_filenames,valid_labels))
valid_dataset = valid_dataset.map(_valid_parse_function)
valid_dataset = valid_dataset.repeat().batch(valid_num)

iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
next_element = iterator.get_next()

train_init_op = iterator.make_initializer(train_dataset)
valid_init_op = iterator.make_initializer(valid_dataset)

# transfer learn
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, channels])
outputs = tf.placeholder(tf.float32, [None, img_class])
valids = tf.placeholder(tf.float32, [None, img_class])
is_training = tf.placeholder(tf.bool)

model = nets.DenseNet201(inputs, is_training=True, classes=img_class, stem=True)
with tf.variable_scope('top_layer'):
    top_layer = tf.reduce_mean(model, [1, 2], name='avgpool')
    if droprate > 0:
        top_layer = tf.layers.dropout(top_layer, droprate)
    if fc_mid_layers > 0:
        top_layer = tf.contrib.layers.fully_connected(top_layer, fc_mid_layers)
    top_layer = tf.contrib.layers.fully_connected(top_layer, img_class)
    top_layer = tf.contrib.layers.softmax(top_layer)

    loss = tf.losses.softmax_cross_entropy(outputs, top_layer)
    acc = accuracy(outputs, top_layer)
    valid_loss = tf.losses.softmax_cross_entropy(valids, top_layer)
    valid_acc = accuracy(valids, top_layer)
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
    if num_train_vars > 0:
        trainable_vars = tf.trainable_variables()
        train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=trainable_vars[-num_train_vars:])
    else:
        train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

loss_scalar = tf.summary.scalar('train_loss', loss, collections=['train'])
acc_scalar = tf.summary.scalar('train_accuracy', acc, collections=['train'])
train_merged = tf.summary.merge_all('train') #train_merged = tf.summary.merge([loss_scalar, acc_scalar])

valid_loss_scalar = tf.summary.scalar('valid_loss', valid_loss, collections=['valid'])
valid_acc_scalar = tf.summary.scalar('validation_accuracy', valid_acc, collections=['valid'])
valid_merged = tf.summary.merge_all('valid') #valid_merged = tf.summary.merge([valid_loss_scalar, valid_acc_scalar])
# merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(summary_train_dir)
valid_writer = tf.summary.FileWriter(summary_valid_dir)
top_layer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='top_layer')
init = tf.variables_initializer(top_layer_vars)#init = tf.global_variables_initializer()

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
saver = tf.train.Saver(max_to_keep=3)

with tf.Session() as sess:

    if status == 'train':
        # sess.run(sess.graph.get_tensor_by_name('beta1_power/Assign:0'))
        # sess.run(sess.graph.get_tensor_by_name('beta2_power/Assign:0'))
        sess.run(init)
        sess.run(model.pretrained())
    elif status == 'restore':
        saver.restore(sess, '../' + model_name + '/model.ckpt')
        print(model_name + ' model restore sucess')

    sess.run(valid_init_op)
    x_valid, y_valid = sess.run(next_element)
    x_valid = model.preprocess(x_valid)

    for n in range(num_epochs):
        print('%d/%d epochs' % (n, num_epochs))
        sess.run(train_init_op)
        num_iters = train_num // batch_size + 1

        for i in range(num_iters):
            x_train, y_train =sess.run(next_element)
            x_train = model.preprocess(x_train)
            sess.run(train, {inputs: x_train, outputs: y_train})
            train_preds, summary_train =  sess.run([top_layer, train_merged], {inputs: x_train, outputs: y_train})
            valid_preds, summary_valid =  sess.run([top_layer, valid_merged], {inputs: x_valid, valids: y_valid})
            train_writer.add_summary(summary_train, n*num_iters + i)
            valid_writer.add_summary(summary_valid, n*num_iters + i)
            print('  %d/%d iteration' % ((i+1), num_iters))

            train_preds = np.argmax(train_preds, axis=1)
            valid_preds = np.argmax(valid_preds, axis=1)
            train_true = np.argmax(y_train, axis=1)
            valid_acc =  np.mean(valid_preds == valid_name['label'].values)
            print('train accuracy:', np.mean(train_preds == train_true))
            print('validation accuracy:', valid_acc)

        valid_preds = sess.run(top_layer, {inputs: x_valid})
        valid_preds = np.argmax(valid_preds, axis=1)
        valid_acc =  np.mean(valid_preds == valid_name['label'].values)
        print('valdation accuracy now is %.2f' % (valid_acc * 100))

        if  valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            print('Best valdation accuracy now is %.2f Model saved' % (best_valid_acc * 100))
            save_path = saver.save(sess, '../' + model_name + '/model.ckpt')
