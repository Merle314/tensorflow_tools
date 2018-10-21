import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def rot90(tensor, k=1, axes=[0, 1], name=None):
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")
    tenor_shape = (tensor.get_shape().as_list())
    dim = len(tenor_shape)
    if axes[0] == axes[1] or np.absolute(axes[0] - axes[1]) == dim:
        raise ValueError("Axes must be different.")
    if (axes[0] >= dim or axes[0] < -dim or axes[1] >= dim or axes[1] < -dim):
        raise ValueError("Axes={} out of range for tensor of ndim={}.".format(
            axes, dim))
    k %= 4
    if k == 0:
        return tensor
    if k == 2:
        img180 = tf.reverse(
            tf.reverse(tensor, axis=[axes[0]]), axis=[axes[1]], name=name)
        return img180

    axes_list = np.arange(0, dim)
    (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]],
                                                axes_list[axes[0]])  # 替换

    print(axes_list)
    if k == 1:
        img90 = tf.transpose(
            tf.reverse(tensor, axis=[axes[1]]), perm=axes_list, name=name)
        return img90
    if k == 3:
        img270 = tf.reverse(
            tf.transpose(tensor, perm=axes_list), axis=[axes[1]], name=name)
        return img270

if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.int32, shape=[3, 3, 1])
    a1 = rot90(a, k=2)
    a2 = rot90(a, k=1)
    a3 = rot90(a, k=3)
    a4 = tf.identity(a)
    ax = tf.concat([a1, a2, a3, a4], axis=2)
    X = tf.reshape(ax, [3, 3, 2, 2])
    X = tf.transpose(X, [0, 2, 1, 3])
    X = tf.reshape(X, [6, 6])
    n = [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]
    with tf.Session() as sess:
        out = sess.run(X, {a:n})
    print(out)
