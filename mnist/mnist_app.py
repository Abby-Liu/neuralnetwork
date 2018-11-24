# coding:utf-8
 
import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward


def restore_model(testPicArr):
 
    with tf.Graph().as_default() as tg:
        # 輸入占位
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        # 輸出占位
        y = mnist_forward.forward(x, None)
        # predict
        preValue = tf.argmax(y, 1)
        # 滑動模型
        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
 
        with tf.Session() as sess:
            # loading checkpoint
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # feed one graph every step
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1
 
 
def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert())
    threshold = 50
    # Training imaged are black-background and white-line
    # , so we have to convert it to white-background and black-line
    # and 2-value handling
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 255
            else:
                im_arr[i][j] = 0
    # convert 2-dimension to 1-dimension
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    # regularization
    img = np.multiply(nm_arr, 1.0 / 255.0)
 
    return nm_arr  # img
 
 
def application():
    for i in range(10):
        testPic = './testData/'+str(i)+'.jpg' # path
        # print(testPic)
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print(i, "The prediction number is ", preValue)

def main():
    application()

if __name__ == '__main__':
    main()