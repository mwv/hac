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
                 spectral):
        """

        Arguments:
        :param codebooks: list of Codebook objects
        :param spectral: Spectral object.
        """
        self.codebooks = codebooks
        self.spectral = spectral
        self._fcs = [0] + list(np.cumsum([cdb.n_features
                                          for cdb in codebooks]))
        self.n_features = self._fcs[-1]

    def convert_sig(self, sig, lags=None):
        """

        Arguments:
        :param sig:
        """
        if lags is None:
            lags = [2, 5]
        f = self._fcs
        spec = np.hstack(self.spectral.transform(sig))
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

    def __init__(self, clf=cluster.KMeans):
        """

        Arguments:
        :param clf:
        :param **kwargs:
        """
        self.clf = clf
        self.trained = False

    def train(self, X):
        """

        Arguments:
        :param X:
        """
        self.clf.fit(X)
        self.n_classes, self.n_features = self.clf.cluster_centers_.shape
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
        inds = self.clf.predict(X)
        t = np.zeros((X.shape[0], self.n_classes), dtype=np.int)
        t[np.arange(X.shape[0]), inds] = 1
        return t
