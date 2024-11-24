import sys
import math
import tensorflow as tf
import math
import numpy as np
import os
import logging, os

logging.disable(logging.WARNING)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')


# zad 4

def terminal_solution():
    args = sys.argv[1:]

    n_args = len(args)

    a, b, c = 1, 1, -n_args

    delta = b * b - 4 * a * c

    n = (-b + math.sqrt(delta)) / 2.0

    if not n.is_integer() and delta >= 0:
        print("wrong amount of numbers")
        return

    n = int(n)

    print("\n", n)
    matx = list(map(float, args[0:n * n]))
    vect = list(map(float, args[n * n:]))

    print(f"Matrix: {matx}")
    print(f"Vec: {vect}")

    matx = tf.constant(matx, shape=[n, n], dtype=tf.float32)
    vect = tf.constant(vect, shape=[n, 1], dtype=tf.float32)

    try:
        inv_matrix = tf.linalg.inv(matx)
    except Exception:
        print("solution can not be found.")
        return

    solve = tf.matmul(inv_matrix, vect)
    print("Solve:")
    idx = 1
    for val in solve.numpy():
        print(f"x{idx}: ", val[0])
        idx += 1


terminal_solution()
