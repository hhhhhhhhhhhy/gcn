"""
初始化器模块，提供不同的权重初始化函数。
这些函数用于在构建神经网络时初始化模型的权重。
"""
import tensorflow as tf
import numpy as np

def uniform(shape, scale=0.05, name=None):
    """
    均匀分布初始化器。

    生成一个在 [--scale, scale] 范围内均匀分布的随机数初始化器。

    参数: 
    shape (tuple): 变量的形状。
    scale (float): 均匀分布的范围，默认为 0.05。
    name (str): 变量的名称，默认为 None。

    返回:
    tf.Variable: 一个均匀分布的 TensorFlow 变量。
    """
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """
    Glorot & Bengio (AISTATS 2010) 初始化器。

    生成一个 Glorot & Bengio 初始化器，通常用于深度学习模型中。

    参数:
    shape (tuple): 变量的形状。
    name (str): 变量的名称，默认为 None。

    返回:
    tf.Variable: 一个 Glorot & Bengio 初始化的 TensorFlow 变量。
    """
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """
    全零初始化器。

    生成一个全零的初始化器。

    参数:
    shape (tuple): 变量的形状。
    name (str): 变量的名称，默认为 None。

    返回:
    tf.Variable: 一个全零的 TensorFlow 变量。
    """
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """
    全一初始化器。

    生成一个全一的初始化器。

    参数:
    shape (tuple): 变量的形状。
    name (str): 变量的名称，默认为 None。

    返回:
    tf.Variable: 一个全一的 TensorFlow 变量。
    """
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
