# -*- coding: utf-8 -*-

import logging
from logging import getLogger, Formatter, StreamHandler, DEBUG
import os
import sys

import numpy as np

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


