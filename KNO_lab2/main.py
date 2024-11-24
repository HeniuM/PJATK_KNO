import sys

import tensorflow as tf
import numpy as np
import math


# Zad 1-2
def rotation(x, y, angle):
    cos_angle = math.cos(angle * math.pi)
    sin_angle = math.sin(angle * math.pi)

    rotation_matrix = tf.constant([cos_angle, -sin_angle, sin_angle, cos_angle]
                                  , shape=[2, 2], dtype=tf.float32)

    point = tf.constant([x, y], shape=[2, 1], dtype=tf.float32)

    return tf.matmul(rotation_matrix, point)


print("Point rotation: \n", rotation(1.0, 0.0, 0.5), "\n")

print("---------------------------------------------- \n")


# Zad 3

def solve_linear(a, b):
    a_inv = tf.linalg.inv(a)
    return tf.matmul(a_inv, b)


a = tf.constant([[3, 1, 1, 2]], dtype=tf.float32, shape=[2, 2])
b = tf.constant([9, 8], dtype=tf.float32, shape=[2, 1])

print("linear function: \n", solve_linear(a, b), "\n")

print("----------------------------------- \n")


#