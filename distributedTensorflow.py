import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time

import tensorflow as tf
from model import leNet, losses, training, evaluate
from data_helper import batch_generation


def main(args):
    task_index = args.task_index
    job_name = args.job_name
    checkpoint_dir = args.checkpoint_dir

    ps_spec = ["localhost:2220"]
    worker_spec = ["localhost:2221", "localhost:2222", "localhost:2223"]
    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    if job_name == "ps":
        server.join()

    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % task_index, cluster=cluster)):
        global_step = tf.train.get_or_create_global_step()
        images_batch, labels_batch = batch_generation(batch_size=4, dataset="training")
        train_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x")
        train_y = tf.placeholder(tf.int32, shape=[None], name="y")

        sotfmax = leNet(train_x, 10, training=args.mode)
        loss = losses(sotfmax, train_y)
        train_op = training(loss, lr=0.001, global_step=global_step)
        acc = evaluate(sotfmax, train_y)
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

    train_writer = tf.summary.FileWriter("TensorBoard" % worker_spec, graph=tf.get_default_graph(),
                                         filename_suffix="train")

    is_chief = (task_index == 0)
    cheif_only_hooks = [tf.train.CheckpointSaverHook(checkpoint_dir, save_steps=100, saver=saver)]
    # hooks = [tf.train.StopAtStepHook(num_steps=args.steps)]

    with tf.train.MonitoredTrainingSession(
            master=server.target,
            is_chief=is_chief,
            scaffold=tf.train.Scaffold(init_op=init_op, summary_op=summary_op, saver=saver),
            checkpoint_dir=checkpoint_dir,
            chief_only_hooks=cheif_only_hooks) as sess:

        step = sess.run(global_step)
        print("Session ready")
        while step < args.steps:
            # The value of a feed cannot be a tf.Tensor object. Acceptable feed values include Python scalars,
            # strings, lists, numpy ndarrays, or TensorHandles. For reference, the tensor object was Tensor(
            # "Reshape:0", shape=(4, 28, 28, 1), dtype=float32, device=/job:worker/task:0)
            images, labels = sess.run([images_batch, labels_batch])
            train_fed = {train_x: images, train_y: labels}
            _, accuracy, cost, summary, step = sess.run([train_op, acc, loss,
                                                         summary_op, global_step],
                                                        feed_dict=train_fed)

            if (step % 1 == 0) and (step < args.steps):
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (
                    step, cost, accuracy * 100.0))
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
                                                                          ["softmax_linear"])
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
