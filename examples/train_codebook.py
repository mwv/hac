#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: train_codebook.py
# date: Fri July 26 18:36:10 2013
# author:
# Maarten Versteegh
# cls.ru.nl/~versteegh
# maartenversteegh AT gmail DOT com
# Centre for Language Studies
# Radboud University Nijmegen
#
# Licensed under GPLv3
# ------------------------------------
"""train_codebook: Train a HAC codebook from a directory of audio files.

Usage:
<code>
train_codebook.py <audiodirectory> <outputfile>
</code>

This will save three codebooks in \c outputfile, one each for the static
MFCC features, their deltas and double-deltas.

The function \c train_codebook can be used separately. It takes a directory,
a spectral feature extraction function and a list of clustering methods as
input arguments and return the codebooks. See the documentation of
\c train_codebook for more information.

"""

from __future__ import division

import os
import fnmatch
import cPickle as pickle

import numpy as np
from sklearn import cluster
from scikits import audiolab, samplerate

from hac import HAC, Codebook
from spectral import MFCC


def rms(X):
    """Root mean square of X

    Arguments:
    :param X: ndarray
    """
    return np.sqrt(np.sum(X**2)/X.shape[0])


def trim_silence(sig, fs, window=25, threshold=0.01, tolerance=100):
    """Return indices of first and last samples

    Arguments:
    :param sig: signal :: ndarray(1,)
    :param fs: samplerate
    :param window: size of window in ms
    :param threshold: energy threshold
    :param tolerance: buffer on both sides of trimmed segment in ms
    """
    winlen = int(window * fs / 1000)
    r = []
    for start in xrange(0, sig.shape[0], winlen//2):
        if start + winlen < sig.shape[0]:
            r.append(rms(sig[start: start + winlen]))
    r = np.array(r)
    ab = np.nonzero(r > threshold)[0]
    tol_f = int(tolerance * fs / 1000)
    return max(0, int(ab[0] * winlen // 2) - tol_f), \
        min(sig.shape[0], int(ab[-1] * winlen // 2) + tol_f)


def rglob(rootdir, pattern):
    """Recursive iglob

    Arguments:
    :param rootdir:
    :param pattern:
    """
    for root, _, files in os.walk(rootdir):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                yield os.path.join(root, basename)


def train_codebook(basedirectory,
                   spectral,
                   desired_fs,
                   clfs,
                   n_samples):
    """Train the codebooks.

    Arguments:
    :param basedirectory: root directory of the audio corpus
    :param spectral:
      Spectral feature extraction.
      Object should be picklable and implement the
      \c Spectral abc; i.e. provide a \c transform method.
    :param clfs:
      list of clusterers. valid clusterers have a \c fit method
      and a \c predict method. optionally, for soft vq, also implement
      a \c predict_proba method.
    :param n_samples:
      number of spectral frames to sample from the audio corpus.
    :returns:
      a list of Codebook objects, of same length as the output of spectral_func
    """
    wavs = list(rglob(basedirectory, '*.wav'))
    np.random.shuffle(wavs)

    inds = None
    idx = 0
    X = None
    for i, wav in enumerate(wavs):
        if i % 10 == 0 and i > 0:
            print 'samples: {3}/{4}; loading file: {0} ({1}/{2})'.format(
                wavs[i],
                i+1,
                len(wavs),
                X.shape[0],
                n_samples
            )
        sig, fs, _ = audiolab.wavread(wav)
        start, stop = trim_silence(sig, fs)
        specs = spectral.transform(samplerate.resample(sig[start:stop],
                                                       desired_fs/fs,
                                                       'sinc_best'))
        if inds is None:
            inds = [0] + list(np.cumsum([spec.shape[1] for spec in specs]))
        spec = np.hstack(specs)
        if idx + spec.shape[0] >= n_samples:
            spec = spec[:n_samples - idx, :]
        if X is None:
            X = spec
        else:
            X = np.vstack((X, spec))
        idx += spec.shape[0]
        if idx >= n_samples:
            break

    cdbs = [Codebook(clf) for clf in clfs]
    for i, cdb in enumerate(cdbs):
        cdb.train(X[:, inds[i]:inds[i+1]])
    return cdbs

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print 'Usage: train_codebook.py <audio directory> <outputfile[.pkl]>'
    n_samples = 1000
    spectral = MFCC()
    fs = spectral.config['fs']
    clfs = [cluster.KMeans(init='k-means++',
                           n_clusters=150,
                           n_init=10,
                           n_jobs=2,
                           random_state=np.random.RandomState(42)),
            cluster.KMeans(init='k-means++',
                           n_clusters=150,
                           n_init=10,
                           n_jobs=2,
                           random_state=np.random.RandomState(42)),
            cluster.KMeans(init='k-means++',
                           n_clusters=100,
                           n_init=10,
                           n_jobs=2,
                           random_state=np.random.RandomState(42))]
    cdbs = train_codebook(sys.argv[1], spectral, fs, clfs, n_samples)
    with open(sys.argv[2], 'wb') as f:
        pickle.dump(HAC(cdbs, spectral), f, -1)
