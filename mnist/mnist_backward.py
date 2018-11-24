#coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_generateds
import os

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME="mnist_model"
train_num_examples = 60000 #2

# Define backward propagation method including regularization
# def backward(mnist):
def backward():

    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE]) 
    y = mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))  # Computes softmax cross entropy cost and gradients to backpropagate.
    cem = tf.reduce_mean(ce)    # Get mean of cross entropy
    loss = cem + tf.add_n(tf.get_collection('losses')) # For regularization

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        # mnist.train.num_examples / BATCH_SIZE, 
        train_num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY,
        staircase=True)

    # Define backward propagation method including regularization
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Setting moving average dynamically 動態設置滑動平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):     # To control flow graph, operate [train_step, ema_op] first
        train_op = tf.no_op(name='train')    # Do nothing

    saver = tf.train.Saver() # Store training parameters
    img_batch, label_batch = mnist_generateds.get_tfRecord(BATCH_SIZE, isTrain=True)    # 3 For new dataset


    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        # 斷點續訓 Restore session
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator() #4
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) # 5

        for i in range(STEPS):
            xs, ys = sess.run([img_batch, label_batch]) # 6
            # xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        coord.request_stop() # 7
        coord.join(threads) # 8

def main():
    # mnist = input_data.read_data_sets("./data/", one_hot=True)
    # backward(mnist)
    backward() # 9

if __name__ == '__main__':
    main()


