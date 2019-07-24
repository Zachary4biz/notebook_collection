import tensorflow as tf
from ops import *


# def mobilenetv2(inputs, num_classes, is_train=True, reuse=False):
#     exp = 6  # expansion ratio
#     with tf.variable_scope('mobilenetv2'):
#         net = conv2d_block(inputs, 32, 3, 2, is_train, name='conv1_1')  # size/2

#         net = res_block(net, 1, 16, 1, is_train, name='res2_1')

#         net = res_block(net, exp, 24, 2, is_train, name='res3_1')  # size/4
#         net = res_block(net, exp, 24, 1, is_train, name='res3_2')

#         net = res_block(net, exp, 32, 2, is_train, name='res4_1')  # size/8
#         net = res_block(net, exp, 32, 1, is_train, name='res4_2')
#         net = res_block(net, exp, 32, 1, is_train, name='res4_3')

#         net = res_block(net, exp, 64, 1, is_train, name='res5_1')
#         net = res_block(net, exp, 64, 1, is_train, name='res5_2')
#         net = res_block(net, exp, 64, 1, is_train, name='res5_3')
#         net = res_block(net, exp, 64, 1, is_train, name='res5_4')

#         net = res_block(net, exp, 96, 2, is_train, name='res6_1')  # size/16
#         net = res_block(net, exp, 96, 1, is_train, name='res6_2')
#         net = res_block(net, exp, 96, 1, is_train, name='res6_3')

#         net = res_block(net, exp, 160, 2, is_train, name='res7_1')  # size/32
#         net = res_block(net, exp, 160, 1, is_train, name='res7_2')
#         net = res_block(net, exp, 160, 1, is_train, name='res7_3')

#         net = res_block(net, exp, 320, 1, is_train, name='res8_1', shortcut=False)

#         net = pwise_block(net, 1280, is_train, name='conv9_1')
#         net = global_avg(net)
#         logits = flatten(conv_1x1(net, num_classes, name='logits'))

#         pred = tf.nn.softmax(logits, name='prob')
#         return logits, pred

# original
def mobilenetv2(inputs, num_classes, is_train=True, reuse=False):
    exp = 6  # expansion ratio
    with tf.variable_scope('mobilenetv2'):
        net = conv2d_block(inputs, 32, 3, 2, is_train, name='conv1_1')  # size/2,,,,32

        net = res_block(net, 1, 16, 1, is_train, name='res2_1')

        net = res_block(net, exp, 24, 2, is_train, name='res3_1')  # size/4
        net = res_block(net, exp, 24, 1, is_train, name='res3_2')

        net = res_block(net, exp, 32, 2, is_train, name='res4_1')  # size/8
        net = res_block(net, exp, 32, 1, is_train, name='res4_2')
        net = res_block(net, exp, 32, 1, is_train, name='res4_3')

        net = res_block(net, exp, 64, 1, is_train, name='res5_1')
        net = res_block(net, exp, 64, 1, is_train, name='res5_2')
        net = res_block(net, exp, 64, 1, is_train, name='res5_3')
        net = res_block(net, exp, 64, 1, is_train, name='res5_4')

        net = res_block(net, exp, 96, 2, is_train, name='res6_1')  # size/16
        net = res_block(net, exp, 96, 1, is_train, name='res6_2')
        net = res_block(net, exp, 96, 1, is_train, name='res6_3')

        net = res_block(net, exp, 160, 2, is_train, name='res7_1')  # size/32
        net = res_block(net, exp, 160, 1, is_train, name='res7_2')
        net = res_block(net, exp, 160, 1, is_train, name='res7_3')

        net = res_block(net, exp, 320, 1, is_train, name='res8_1', shortcut=False)

        net = pwise_block(net, 1280, is_train, name='conv9_1')
        net = global_avg(net)
        logits = flatten(conv_1x1(net, num_classes, name='logits'))

        pred = tf.nn.softmax(logits, name='prob')
        return logits, pred
    
# clip
def mobilenetv2_caffe(inputs, num_classes, is_train=True, reuse=False):
    exp = 6  # expansion ratio
    with tf.variable_scope('mobilenetv2'):
        net = conv2d_block(inputs, 16, 3, 2, is_train, name='conv1_1')  # size/2,,,,32

        # net = res_block(net, 1, 16, 1, is_train, name='res2_1')

        net = res_block(net, exp, 24, 2, is_train, name='res3_1')  # size/4
        net = res_block(net, exp, 24, 1, is_train, name='res3_2')

        net = res_block(net, exp, 32, 2, is_train, name='res4_1')  # size/8
        net = res_block(net, exp, 32, 1, is_train, name='res4_2')
        # net = res_block(net, exp, 32, 1, is_train, name='res4_3')

        net = res_block(net, exp, 64, 2, is_train, name='res5_0')
        net = res_block(net, exp, 64, 1, is_train, name='res5_1')
        # net = res_block(net, exp, 64, 1, is_train, name='res5_2')
        # net = res_block(net, exp, 64, 1, is_train, name='res5_3')
        # net = res_block(net, exp, 64, 1, is_train, name='res5_4')

        # net = res_block(net, exp, 96, 2, is_train, name='res6_1')  # size/16
        # net = res_block(net, exp, 96, 1, is_train, name='res6_2')
        # net = res_block(net, exp, 96, 1, is_train, name='res6_3')

        # net = res_block(net, exp, 160, 2, is_train, name='res7_1')  # size/32
        # net = res_block(net, exp, 160, 1, is_train, name='res7_2')
        # net = res_block(net, exp, 160, 1, is_train, name='res7_3')

        # net = res_block(net, exp, 320, 1, is_train, name='res8_1', shortcut=False)

        # net = pwise_block(net, 1280, is_train, name='conv9_1')
        # net = global_avg(net)
        # logits = flatten(conv_1x1(net, num_classes, name='logits'))

        net = tf.layers.flatten(inputs=net)
        net = tf.layers.dense(
            inputs=net,
            units=200,
            activation=None,
            use_bias=True,
            name="fc1")

        net = tf.layers.dense(
            inputs=net,
            units=200,
            activation=None,
            use_bias=True,
            name="fc2")

        net = tf.layers.dense(
            inputs=net,
            units=50,
            activation=None,
            use_bias=True,
            name="fc3")

        logits = tf.layers.dense(
            inputs=net,
            units=num_classes,
            activation=None,
            use_bias=True,
            name="fc4")

        pred = tf.nn.softmax(logits, name='prob')
        return logits, pred
