# coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import cifar_backward
import cifar_forward
import sys
import os

# Restore model and predict
def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, cifar_forward.INPUT_NODE])
        y = cifar_forward.forward(x, None)
        preValue = tf.argmax(y, 1) # The maximum label is the prediction

        variable_averages = tf.train.ExponentialMovingAverage(cifar_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            # Restore model
            ckpt = tf.train.get_checkpoint_state(cifar_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                # Predict
                preValue = sess.run(preValue, feed_dict={x:testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1

# Picture preprocessing
def pre_pic(picName):
    img = Image.open(picName)
    # 重新设置图片的尺寸
    reIm = img.resize((32, 32), Image.ANTIALIAS)
    im_arr = np.array(reIm)

    nm_arr = im_arr.reshape([1, 1024*3])
    nm_arr = nm_arr.astype(np.float32)
    # regulariztion
    img = np.multiply(nm_arr, 1.0 / 255.0)
 
    return nm_arr  # img

def application():
    train_dir = '../cifar-10/'
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    for label in labels:
        testPic = './testData/'+label+'.jpg' # path
        testPicArr = pre_pic(testPic) # Preprocess pictures

        preClass = restore_model(testPicArr) # Predict pictures
        print('Image:', testPic, end='\n')
        print('The prediction object is ', labels[int(preClass)])

def main():
    application()
if __name__ == '__main__':
    main()