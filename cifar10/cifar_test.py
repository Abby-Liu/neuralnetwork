#coding:utf-8

import time
import tensorflow as tf
import cifar_forward
import cifar_backward
import cifar_generateds
TEST_INTERVAL_SECS = 5
TEST_NUM = 10000 #1

def test():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, cifar_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, cifar_forward.OUTPUT_NODE])
        y = cifar_forward.forward(x, None)

        # set moving average dynamically
        ema = tf.train.ExponentialMovingAverage(cifar_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()    # 將影子變量直接映射到變量本身
        saver = tf.train.Saver(ema_restore)     # 使用之前保存的訓練參數並保存起來

        # comput corret rate
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        img_batch, label_batch = cifar_generateds.get_tfRecord(TEST_NUM, isTrain=False)    # 2 For new dataset

        while True:
            with tf.Session() as sess:
                # Restore session
                ckpt = tf.train.get_checkpoint_state(cifar_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path) # 使用之前保存的訓練參數
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    
                    coord = tf.train.Coordinator() # 3
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord) # 4

                    xs, ys = sess.run([img_batch, label_batch]) # 5

                    accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))

                    coord.request_stop() # 6
                    coord.join(threads) # 7
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    test()

if __name__ == '__main__':
    main()
