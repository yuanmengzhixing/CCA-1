# -*- coding: utf-8 -*-

import os

import numpy as np

import utils

def load_mnist(split):
    """
    :type split: str
    :rtype: numpy.ndarray(shape=(N,28*14), dtype=np.float32), numpy.ndarray(shape=(N,28*14), dtype=np.float32)
    """
    config = utils.Config()

    path_images = os.path.join(config.getpath("mnist"), "images.%s.txt" % split)
    images = load_images(path_images)
    N = len(images)
    vectors_left = images[:,:,:14] # NOTE
    vectors_right = images[:,:,14:] # NOTE
    vectors_left = vectors_left.reshape(N, -1)
    vectors_right = vectors_right.reshape(N, -1)
    return vectors_left, vectors_right

def load_mnist2(split):
    """
    :type split: str
    :rtype: numpy.ndarray(shape=(N,14*28), dtype=np.float32), numpy.ndarray(shape=(N,14*28), dtype=np.float32)
    """
    config = utils.Config()

    path_images = os.path.join(config.getpath("mnist"), "images.%s.txt" % split)
    images = load_images(path_images)
    N = len(images)
    vectors_top = images[:,:14,:] # NOTE
    vectors_bottom = images[:,14:,:] # NOTE
    vectors_top = vectors_top.reshape(N, -1)
    vectors_bottom = vectors_bottom.reshape(N, -1)
    return vectors_top, vectors_bottom

def load_images(path):
    """
    :type path: str
    :rtype: numpy.ndarray of shape (N, 28, 28) and dtype=np.float32
    """
    images = []
    for line in open(path):
        vector = line.decode("utf-8").strip().split()
        vector = [float(x) for x in vector]
        vector = np.asarray(vector, dtype=np.float32)
        assert vector.shape == (28*28,)
        image = vector.reshape(28, 28)
        images.append(image)
    images = np.asarray(images, dtype=np.float32)
    return images





