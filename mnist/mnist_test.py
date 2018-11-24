#coding:utf-8

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
TEST_INTERVAL_SECS = 5
TEST_NUM = 10000 #1

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()    # 將影子變量直接映射到變量本身
        saver = tf.train.Saver(ema_restore)     # 使用之前保存的訓練參數並保存起來

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        img_batch, label_batch = mnist_generateds.get_tfrecord(TEST_NUM, isTrain=False)    # 2 For new dataset

        while True:
            with tf.Session() as sess:
                # 斷點續訓 Restore session
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path) # 使用之前保存的訓練參數
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    
                    coord = tf.train.Coordinator() # 3
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord) # 4

                    xs, ys = sess.run([img_batch, label_batch]) # 5

                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))

                    coord.request_stop() # 6
                    coord.join(threads) # 7
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()
