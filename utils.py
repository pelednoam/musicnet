import multiprocessing
import os
import numpy as np
import os.path as op
try:
    import cPickle as pickle
except:
    import pickle


def save(obj, fname):
    with open(fname, 'wb') as fp:
        pickle.dump(obj, fp, protocol=4)


def load(fname):
    with open(fname, 'rb') as fp:
        obj = pickle.load(fp)
    if obj is None:
        print('the data in {} is None!'.format(fname))
    return obj


def merge_dictionaries(dict1, dict2):
    return {**dict1, **dict2}


def run_parallel(func, params, njobs=1):
    if njobs == 1:
        results = []
        for run, p in enumerate(params):
            results.append(func(p))
    else:
        pool = multiprocessing.Pool(processes=njobs)
        results = pool.map(func, params)
        pool.close()
    return results


def make_dir(fol):
    if not op.isdir(fol):
        os.makedirs(fol)
    return fol


def add_annotation(text, x, y):
    import pylab
    pylab.annotate(
        text, xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


def calc_windows(data_len, windows_length, windows_shift):
    import math
    if windows_length == 0:
        windows_length = data_len
        windows_num = 1
    else:
        windows_num = math.floor((data_len - windows_length) / windows_shift + 1)
    windows = np.zeros((windows_num, 2))
    for win_ind in range(windows_num):
        windows[win_ind] = [win_ind * windows_shift, win_ind * windows_shift + windows_length]
    windows = windows.astype(np.int)
    return windows
