# -*- coding: utf-8 -*-

import argparse
import os
import sys

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.externals import joblib

import dataloader
import utils

VAL_SIZE = 10000

def evaluate(cca, vectors1, vectors2):
    """
    :type cca: sklearn.cross_decomposition.CCA
    :type vectors1: numpy.ndarray(shape=(N,D1), dtype=np.float32)
    :type vectors2: numpy.ndarray(shape=(N,D1), dtype=np.float32)
    :rtype: float
    """
    Y1, Y2 = cca.transform(vectors1, vectors2) # (N, D), (N, D)
    Y1 = Y1 / np.linalg.norm(Y1, axis=1)[:,None] # (N, D)
    Y2 = Y2 / np.linalg.norm(Y2, axis=1)[:,None] # (N, D)
    sim_mat = np.dot(Y1, Y2.T) # (N, N)

    corr = np.sum(np.diag(sim_mat)) / float(len(sim_mat))

    mean_rank = 0.0
    for data_i in range(len(sim_mat)):
        sim_vec = sim_mat[data_i]
        order = (-sim_vec).argsort()
        ranks = order.argsort()
        ranks += 1
        mean_rank += ranks[data_i]
    mean_rank /= float(len(sim_mat))

    return corr, mean_rank

def main(args):
    # Arguments
    dataset_name = args.dataset
    path_config = args.config
    trial_name = args.name
    mode = args.mode

    utils.logger.debug("[args] dataset=%s" % dataset_name)
    utils.logger.debug("[args] config=%s" % path_config)
    utils.logger.debug("[args] name=%s" % trial_name)
    utils.logger.debug("[args] mode=%s" % trial_name)

    # Preparation
    config = utils.Config(path_config)
    basename = utils.get_basename(dataset_name=dataset_name,
                                  path_config=path_config,
                                  trial_name=trial_name)
    path_log = os.path.join(config.getpath("log"), basename + ".log")
    path_snapshot = os.path.join(config.getpath("snapshot"), basename + ".model.pkl.cmp")
    path_eval = os.path.join(config.getpath("evaluation"), basename + ".txt")
    path_anal = os.path.join(config.getpath("analysis"), basename)

    utils.logger.debug("[path] snapshot=%s" % path_snapshot)
    if mode == "train":
        utils.logger.debug("[path] log=%s" % path_log)
        utils.set_logger(path_log)
    elif mode == "evaluation":
        utils.logger.debug("[path] evaluation=%s" % path_eval)
        utils.set_logger(path_eval)
    elif mode == "analysis":
        utils.logger.debug("[path] analysis=%s" % path_anal)

    # Data preparation
    if dataset_name == "mnist":
        vectors1_trainval, vectors2_trainval = dataloader.load_mnist(split="train")
        vectors1_train = vectors1_trainval[:-VAL_SIZE]
        vectors2_train = vectors2_trainval[:-VAL_SIZE]
        vectors1_val = vectors1_trainval[-VAL_SIZE:]
        vectors2_val = vectors2_trainval[-VAL_SIZE:]
        vectors1_test, vectors2_test = dataloader.load_mnist(split="test")
    else:
        utils.logger.debug("[data] Error: Unknown dataset_name=%s" % dataset_name)
        sys.exit(-1)
    utils.logger.debug("[data] # of training pairs=%d" % len(vectors1_train))
    utils.logger.debug("[data] # of validation pairs=%d" % len(vectors1_val))
    utils.logger.debug("[data] # of test pairs=%d" % len(vectors1_test))
    utils.logger.debug("[data] Dimensionalities=%d and %d" % \
                            (vectors1_train.shape[1], vectors1_train.shape[1]))

    # Hyper parameters
    shared_dim = config.getint("shared_dim")

    utils.logger.debug("[hyperparams] shared_dim=%d" % shared_dim)

    # Model preparation
    if mode == "train":
        cca = CCA(n_components=shared_dim)
        utils.logger.debug("[model] Initialized the CCA model.")
    else:
        cca = joblib.load(path_snapshot)
        utils.logger.debug("[model] Loaded the CCA model from %s" % path_snapshot)

    # Training, evaluation, or analysis
    if mode == "train":
        utils.logger.debug("[info] Fitting ...")
        cca.fit(vectors1_train, vectors2_train)
        corr, mean_rank = evaluate(cca, vectors1_val, vectors2_val)
        utils.logger.debug("[validation] Correlation coefficients=%f, Mean Rank=%f" % \
                (corr, mean_rank))
        joblib.dump(cca, path_snapshot, compress=True)
        utils.logger.debug("[info] Saved.")
    elif mode == "evaluation":
        # Validation
        corr, mean_rank = evaluate(cca, vectors1_val, vectors2_val)
        utils.logger.debug("[validation] Correlation coefficients=%f, Mean Rank=%f" % \
                (corr, mean_rank))
        # Test
        corr, mean_rank = evaluate(cca, vectors1_test, vectors2_test)
        utils.logger.debug("[test] Correlation coefficients=%f, Mean Rank=%f" % \
                (corr, mean_rank))
    elif mode == "analysis":
        pass

    utils.logger.debug("[info] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()
    main(args)
