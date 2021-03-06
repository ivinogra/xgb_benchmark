import numpy as np
import logging
from time import time
import xgboost as xgb
import argparse
import os
import sys
import gc

def create_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('dmatrix_timing.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = create_logger()

def create_data(args):
    r = np.random.RandomState(args.seed)
    a = r.rand(1<<args.rows, 1<<args.columns).astype(np.float32)
    threshold = 0.3
    return np.where(a > threshold, a, np.nan)

def main(args):
    logger.info('\n----------------------')
    logger.info(sys.version)
    logger.info('xgboost version: {}'.format(xgb.__version__))
    logger.info('rows: {}, columns: {}'.format(args.rows, args.columns))
    if args.cache is None:
        logger.info('Generating data')
        a = create_data(args)
    elif os.path.exists(args.cache):
        logger.info('Loading cached data')
        a = np.load(args.cache)['arr_0']
    else:
        logger.info('Generating and caching data')
        a = create_data(args)
        np.savez_compressed(args.cache, a)

    t = [int(x) for x in args.t]
    for k in t:
        logger.info('Running with {} threads'.format(k))
        stamp = time()
        dm = xgb.DMatrix(a, missing=np.nan, nthread=k)
        logger.info('Using {0} threads; time: {1:0.04f} seconds'.format(k, time() - stamp))
        del dm
        gc.collect()

    logger.info("Done!")


if __name__ == '__main__':
    A = argparse.ArgumentParser(description='DMatrix creating timing')
    A.add_argument('-t', help='nthreads', nargs='+')
    A.add_argument('--cache', help='where to cache the np.array')
    A.add_argument('--seed', help='random seed, ignored if cache is available', type=int)
    A.add_argument('--rows', help='log2 of number of rows in array', type=int, default=22)
    A.add_argument('--columns', help='log2 of number of columns in array', type=int, default=8)
    main(A.parse_args())
