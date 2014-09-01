#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: hac.py
# date: Fri July 26 18:05:55 2013
# author:
# Maarten Versteegh
# cls.ru.nl/~versteegh
# maartenversteegh AT gmail DOT com
# Centre for Language Studies
# Radboud University Nijmegen
#
# Licensed under GPLv3
# ------------------------------------
"""hac: train codebooks and HAC objects

"""

from __future__ import division

from itertools import product

import numpy as np
from numpy import hstack, ravel, outer
from sklearn import cluster


class HAC(object):
    """Primary object for interacting with HAC. Initialize with Codebook and
    Spectral object. Call \c convert_sig to convert
    a signal to HAC representation.
    """

    def __init__(self,
                 codebooks,
                 spectral,
                 lags=None,
                 vq_method='hard'):
        """

        Arguments:
        :param codebooks: list of Codebook objects
        :param spectral: Spectral object.
        :param lags: list of intervals [2,5]
        :param vq_method: vector quantization method ['hard']
        """
        self.codebooks = codebooks
        self.spectral = spectral
        if lags is None:
            lags = [2, 5]
        self.lags = lags
        self.vq_method = vq_method

        self._fcs = [0] + list(np.cumsum([cdb.n_features
                                          for cdb in codebooks]))
        self.n_spec_feats = self._fcs[-1]
        self.n_spec_feats = len(self.lags) * sum([cdb.n_clusters ** 2
                                                  for cdb in codebooks])

    def convert_sig(self, sig):
        """Convert signal to HAC representation.

        Arguments:
        :param sig:
        """
        spec = np.hstack(self.spectral.transform(sig))
        return self.convert_spec(spec)

    def convert_spec(self, spec):
        lags = self.lags
        f = self._fcs

        if spec.shape[1] != self.n_features:
            raise ValueError('Incorrect number of features in spectral coding.'
                             'Got {0} features, expected {1}'.format(
                                 spec.shape[1],
                                 self.n_features))
        r = hstack(np.sum(ravel(outer(t[i], t[i+lag]), order='F')
                          for i in range(t.shape[0]-lag))
                   for (lag, t) in product(lags,
                                           (x.VQ(spec[:,
                                                      f[j]:f[j+1]])
                                            for j, x in
                                            enumerate(self.codebooks))))
        return r


class Codebook(object):
    """
    """

    def __init__(self, clf=cluster.KMeans()):
        """

        Arguments:
        :param clf:
        :param **kwargs:
        """
        self.clf = clf
        if not hasattr(self.clf, 'fit') or not hasattr(self.clf, 'predict'):
            raise TypeError('clf must implement '
                            '`fit` and `predict` methods.')

        if isinstance(clf, cluster.KMeans):
            self.n_clusters = clf.n_clusters
        elif isinstance(clf, mixture.GMM):
            self.n_clusters = clf.n_components
        self.trained = False

    def train(self, X):
        """

        Arguments:
        :param X:
        """
        self.clf.fit(X)
        self.n_features = X.shape[1]
        self.trained = True
        return self

    def VQ(self, X, method='hard'):
        """

        Arguments:
        :param X:
        """
        if not self.trained:
            raise ValueError('Cannot do VQ on untrained codebook')
        if X.shape[1] != self.n_features:
            raise ValueError('Incorrect number of features.'
                             'Got {0} features, expected {1}.'.format(
                                 X.shape[1],
                                 self.n_features))
        if method == 'hard':
            inds = self.clf.predict(X)
            t = np.zeros((X.shape[0], self.n_classes), dtype=np.int)
            t[np.arange(X.shape[0]), inds] = 1
        else:
            if not hasattr(self.clf, 'predict_proba'):
                raise TypeError('clf must implement `predict_proba` method'
                                ' if used with soft vq.')
            t = self.clf.predict_proba(X)
        return t
