# -*- coding: utf-8 -*-

import logging
from logging import getLogger, Formatter, StreamHandler, DEBUG
import os
import shutil
import sys

import numpy as np
from chainer import cuda, optimizers, Variable

###############################
# logging

logger = getLogger("logger")
logger.setLevel(DEBUG)

handler = StreamHandler()
handler.setLevel(DEBUG)
handler.setFormatter(Formatter(fmt="%(message)s"))
logger.addHandler(handler)

def set_logger(filename):
    if os.path.exists(filename):
        logger.debug("[utils.set_logger] A file %s already exists." % filename)
        do_remove = raw_input("[utils.set_logger] Delete the existing log file? [y/n]: ")
        if (not do_remove.lower().startswith("y")) and (not len(do_remove) == 0):
            logger.debug("[utils.set_logger] Done.")
            sys.exit(0)
    logging.basicConfig(level=DEBUG, format="%(message)s", filename=filename, filemode="w")

############################
# NN

def convert_ndarray_to_variable(xs, seq):
    """
    :type xs: numpy.ndarray of shape (N, L)
    :rtype: L-length list of Variable(shae=(N,)) or Variable(shape=(N,L))
    """
    if seq:
        return [Variable(cuda.cupy.asarray(xs[:,j]))
                for j in xrange(xs.shape[1])]
    else:
        return Variable(cuda.cupy.asarray(xs))

def get_optimizer(name="smorms3"):
    """
    :type name: str
    :rtype: chainer.Optimizer
    """
    if name == "adadelta":
        opt = optimizers.AdaDelta()
    elif name == "adam":
        opt = optimizers.Adam()
    elif name == "rmsprop":
        opt = optimizers.RMSprop()
    elif name == "smorms3":
        opt = optimizers.SMORMS3()
    else:
        logger.debug("[utils.get_optimizer;error] Optimizer name %s is not found." % name)
        sys.exit(-1)
    return opt

############################
# others

def mkdir(path, newdir=None):
    """
    :type path: str
    :type newdir: str
    :rtype: None
    """
    if newdir is None:
        target = path
    else:
        target = os.path.join(path, newdir)
    if not os.path.exists(target):
        os.makedirs(target)
        logger.debug("[utils.mkdir] Created a directory=%s" % target)

def get_basename(dataset_name, path_config, trial_name):
    """
    :type dataset_name: str
    :type path_config: str
    :type trial_name: str
    :rtype: str
    """
    basename = "%s.%s.%s" % (dataset_name, get_name(path_config), trial_name)
    return basename

def get_name(path):
    """
    :type path: str
    :rtype: str
    """
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0]

class BestScoreHolder(object):

    def __init__(self, scale=1.0):
        self.best_score = -np.inf
        self.best_step = 0
        self.patience = 0
        self.scale = scale

    def init(self):
        self.best_score = -np.inf
        self.best_step = 0
        self.patience = 0

    def compare_scores(self, score, step):
        if self.best_score <= score:
            # スコア更新
            logger.debug("[best_score,step,patience] (best_score=%.02f, best_step=%d, patience=%d) => (%.02f, %d, %d)" % \
                    (self.best_score * self.scale, self.best_step, self.patience,
                     score * self.scale, step, 0))
            self.best_score = score
            self.best_step = step
            self.patience = 0
            return True
        else:
            # patienceのインクリ
            logger.debug("[best_score,step,patience] (best_score=%.02f, best_step=%d, patience=%d) => (%.02f, %d, %d)" % \
                    (self.best_score * self.scale, self.best_step, self.patience,
                     self.best_score * self.scale, self.best_step, self.patience+1))
            self.patience += 1
            return False

    def ask_finishing(self, max_patience):
        if self.patience >= max_patience:
            return True
        else:
            return False


