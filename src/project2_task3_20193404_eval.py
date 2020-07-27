# EE807 Special Topics in EE <Deep Reinforcement Learning and AlphaGo>, Fall 2019
# Information Theory & Machine Learning Lab, School of EE, KAIST
#
# This is an example code to show how your code should be formatted for task 3
# so that T/A's can perform round-robin tournament among students.
# Written by Jisoo Lee, Su Young Lee, Minguk Jang

import tensorflow as tf
import numpy as np

nx = 6; ny = 6
last_digit = 4

def network(state):
    init_weight = tf.random_normal_initializer(stddev = 0.1)
    init_bias = tf.constant_initializer(0.1)

    # Create variables "weights1" and "biases1".
    weights1 = tf.get_variable("weights1", [3, 3, 3, 30], initializer = init_weight)
    biases1 = tf.get_variable("biases1", [30], initializer = init_bias)

    # Create 1st layer
    conv1 = tf.nn.conv2d(state, weights1, strides = [1, 1, 1, 1], padding = 'SAME')
    out1 = tf.nn.relu(conv1 + biases1)

    # Create variables "weights2" and "biases2".
    weights2 = tf.get_variable("weights2", [3, 3, 30, 50], initializer = init_weight)
    biases2 = tf.get_variable("biases2", [50], initializer = init_bias)

    # Create 2nd layer
    conv2 = tf.nn.conv2d(out1, weights2, strides = [1, 1, 1, 1], padding ='SAME')
    out2 = tf.nn.relu(conv2 + biases2)

    # Create variables "weights3" and "biases3".
    weights3 = tf.get_variable("weights3", [3, 3, 50, 2], initializer = init_weight)
    biases3 = tf.get_variable("biases3", [2], initializer = init_bias)

    # Create 3-1 layer for policy
    conv3 = tf.nn.conv2d(out2, weights3, strides = [1, 1, 1, 1], padding ='SAME')
    out3 = tf.nn.relu(conv3 + biases3)

    # Create 1st fully connected layer
    weights4 = tf.get_variable("weights4", [nx*ny*2, 60+last_digit], initializer = init_weight)
    biases4 = tf.get_variable("biases4", [60+last_digit], initializer = init_bias)
    
    out3_flat = tf.reshape(out3, [-1, nx * ny * 2])
    fc1 = tf.nn.relu(tf.matmul(out3_flat, weights4) + biases4)
    
    # Create 2nd fully connected layer
    weights5 = tf.get_variable("weights5", [60+last_digit, nx * ny], initializer = init_weight)
    biases5 = tf.get_variable("biases5", [nx*ny], initializer = init_bias)
    
    logit = tf.matmul(fc1, weights5) + biases5
    #logit = tf.reshape(out3, [-1, 6 * 6])
    
    # Create 3-2 layer for value
    weightsv1 = tf.get_variable("weightsv1", [3, 3, 50, 1], initializer = init_weight)
    biasesv1 = tf.get_variable("biasesv1", [1], initializer = init_bias)
    
    convv1 = tf.nn.conv2d(out2, weightsv1, strides = [1, 1, 1, 1], padding ='SAME')
    outv1 = tf.nn.relu(convv1 + biasesv1)

    # Create variables "weights1fc" and "biases1fc".
    weightsv2 = tf.get_variable("weightsv2", [nx * ny, 1], initializer = init_weight)
    biasesv2 = tf.get_variable("biasesv2", [1], initializer = init_bias)

    # Create 1st fully connected layer
    fcv1 = tf.reshape(outv1, [-1, nx * ny])
    value = tf.math.tanh(tf.matmul(fcv1, weightsv2) + biasesv2)

    ''' [IMPORTANT] tf.nn.softmax(logit) WILL BE USED FOR GRADING '''
    return (value, logit), tf.nn.softmax(logit, name='softmax')
