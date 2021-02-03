#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf





with tf.Session() as sess:    
    saver = tf.train.import_meta_graph('model.ckpt-0.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    print(sess.run('w1:0'))