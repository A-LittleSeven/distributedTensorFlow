import tensorflow as tf


def leNet(imgs, num_classes, training=True):
    input_layer = tf.reshape(imgs, [-1, 28, 28, 1], name="input_layer")

    conv_1 = tf.layers.conv2d(inputs=input_layer,
                              filters=32,
                              kernel_size=[5, 5],
                              padding="same",
                              activation=tf.nn.relu)

    pool_1 = tf.layers.max_pooling2d(inputs=conv_1, padding=[2, 2], strides=2)

    conv_2 = tf.layers.conv2d(inputs=pool_1,
                              filters=64,
                              kernel_size=[5, 5],
                              padding="same",
                              activation=tf.nn.relu)
    pool_2 = tf.layers.max_pooling2d(inputs=conv_2, padding=[2, 2], strides=2)

    fatten = tf.reshape(pool_2, [-1, 7*7*64])

    dense = tf.layers.dense(inputs=fatten, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=training)

    softmax = tf.layers.dense(inputs=dropout, unit=num_classes, activation=tf.nn.softmax, name="softmax_linear")
    return softmax


def losses(softmax, labels):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax, labels=labels)
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "/loss", loss)
    return loss


def training(loss, lr, global_step):
    with tf.name_scope("optimizer"):
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = opt.minimize(loss, global_step=global_step)
    return train_op


def evaluate(softmax, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(softmax, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "/accuracy", accuracy)
    return accuracy

