import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time

import tensorflow as tf
from model import LeNet, losses, training, evaluation
from data_helper import batch_generation
import numpy as np


def main(args):
    task_index = args.task_index
    job_name = args.job_name
    checkpoint_dir = args.checkpoint_dir

    # one ps and two worker stimulate
    ps_spec = ["localhost:2220"]
    worker_spec = ["localhost:2221"]
    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    if job_name == "ps":
        server.join()

    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % task_index, cluster=cluster)):

        global_step = tf.train.get_or_create_global_step()
        images_batch, labels_batch = batch_generation(batch_size=32, dataset="training")
        # Technically the placeholder doesn't need a shape at all. It can be defined as such.
        # x =tf.placeholder(tf.float32, shape=[])
        # entry for inference.
        train_x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name="x")
        train_y = tf.compat.v1.placeholder(tf.int32, shape=[None], name="y")

        logits = LeNet(train_x, 10, mode=args.mode)
        loss = losses(logits, train_y)
        train_op = training(loss, learning_rate=0.0005, global_step=global_step)
        acc = evaluation(logits, train_y)
        saver = tf.compat.v1.train.Saver()
        summary_op = tf.compat.v1.summary.merge_all()
        init_op = tf.compat.v1.global_variables_initializer()
        print("Variables initialized ...")

    train_writer = tf.compat.v1.summary.FileWriter("TensorBoard" % worker_spec,
                                                   graph=tf.compat.v1.get_default_graph(),
                                                   filename_suffix="train")

    is_chief = (task_index == 0)
    cheif_only_hooks = []
    # cheif_only_hooks = [tf.train.CheckpointSaverHook(checkpoint_dir, save_steps=100, saver=saver)]
    hooks = [tf.train.StopAtStepHook(num_steps=args.steps)]

    with tf.train.MonitoredTrainingSession(
            master=server.target,
            is_chief=is_chief,
            scaffold=tf.train.Scaffold(init_op=init_op, summary_op=summary_op, saver=saver),
            checkpoint_dir=checkpoint_dir,
            hooks=hooks,
            chief_only_hooks=cheif_only_hooks) as sess:

        # add a pesto_feed to feed placeholder during training(hopefully help)
        pesto_fed = {train_x: np.asarray(np.random.rand(784).reshape([1, 784])),
                     train_y: np.asarray([1])}
        # TODO: error feeding data into feed_dict
        step = sess.run(global_step, feed_dict=pesto_fed)
        print("Session ready")
        while not sess.should_stop():
            # The value of a feed cannot be a tf.Tensor object. Acceptable feed values include Python scalars,
            # strings, lists, numpy ndarrays, or TensorHandles. For reference, the tensor object was Tensor(
            # "Reshape:0", shape=(4, 28, 28, 1), dtype=float32, device=/job:worker/task:0)
            tra_images, tra_labels = sess.run([images_batch, labels_batch], feed_dict=pesto_fed)
            train_fed = {train_x: tra_images, train_y: tra_labels}
            _, tra_acc, cost, summary, step = sess.run([train_op, acc, loss,
                                                        summary_op, global_step],
                                                       feed_dict=train_fed)

            if (step % 1 == 0) and (step < args.steps):
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (
                    step, cost, tra_acc * 100.0))
                if task_index == 0:
                    train_writer.add_summary(summary, step)

        print("Stopping MonitoredTrainingSession")

        if is_chief:
            # https://github.com/tensorflow/tensorflow/issues/8425 TypeError: 'sess' must be a Session;
            # <tensorflow.python.training.monitored_session.MonitoredSession object at 0x2be710d0>
            while type(sess).__name__ != 'Session':
                sess = sess._sess
            saver.save(sess,
                       os.path.join(checkpoint_dir, 'model.ckpt'),
                       global_step=global_step)

            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                          ["logits"])
            tf.train.write_graph(constant_graph, checkpoint_dir,
                                 "saved_model_{step}.pb".format(step=step), as_text=False)

        # WORKAROUND FOR https://github.com/tensorflow/tensorflow/issues/21745
        # wait for all other nodes to complete (via done files)
        done_dir = "{}/{}/done".format(checkpoint_dir, 'train')
        print("Writing done file to: {}".format(done_dir))
        tf.gfile.MakeDirs(done_dir)
        with tf.gfile.GFile("{}/{}".format(done_dir, task_index), 'w') as done_file:
            done_file.write("done")

        for i in range(60):
            if len(tf.gfile.ListDirectory(done_dir)) < len(worker_spec):
                print("Waiting for other nodes")
                time.sleep(1)
            else:
                print("All nodes done")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--job_name", type=str, help="ps or worker")
    parser.add_argument("--task_index", type=int, help="define task index")
    parser.add_argument("--checkpoint_dir", type=str, help="like meow/checkpoint", default="./checkpoint")
    parser.add_argument("--mode", type=bool, help="setting mode for model(training/testing)", default=True)
    parser.add_argument("--steps", type=int, help="training Steps", default=1000)

    args = parser.parse_args()

    main(args=args)
