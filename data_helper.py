import os
import time

import numpy as np
import tensorflow as tf


# source: http://yann.lecun.com/exdb/mnist/
def read(dataset="training", data_path=r".\dataset\mnist"):
    if dataset is "training":
        fname_img = os.path.join(data_path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(data_path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(data_path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(data_path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    with open(fname_lbl, 'rb') as flbl:
        label = np.frombuffer(flbl.read(), np.uint8, offset=16)

    with open(fname_img, 'rb') as fimg:
        img = np.frombuffer(fimg.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    return img, label


def batch_generation(batch_size, dataset="training"):
    """
    generate batch from exists data in cycle

    :param batch_size:
    :param dataset:
    :return:
    """
    img, lbl = read(dataset)
    # Normalization
    img = img / 255
    img, lbl = tf.cast(img, tf.float32), tf.cast(lbl, tf.int32)

    # You can add a filename queue, e.g., string_input_producer, to run over all files in the folder with
    # replacement. And try to comment out the shuffle_batch and see if the filename queue is getting any data. This
    # method could run through multiple times if you left num_epoch to none.
    image, label = tf.train.slice_input_producer([img, lbl], capacity=500)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              dynamic_pad=False,  # All rows are the same shape.
                                              enqueue_many=False,  # produces one row at a time.
                                              name='batching')

    label_batch = tf.cast(label_batch, tf.int32)
    return image_batch, label_batch


if __name__ == '__main__':
    # test batch generation
    with tf.device("/cpu:0"):
        print("starting....")
        s = 0
        img_batch, lbl_batch = batch_generation(batch_size=32)
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            # sess.run(tf.initialize_local_variables())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            while True:
                images, label = sess.run([img_batch, lbl_batch])
                time.sleep(1)
                print(images.dtype)
                print(label.dtype)
