#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: sig2hac.py
# date: Fri July 26 22:57:44 2013
# author:
# Maarten Versteegh
# cls.ru.nl/~versteegh
# maartenversteegh AT gmail DOT com
# Centre for Language Studies
# Radboud University Nijmegen
#
# Licensed under GPLv3
# ------------------------------------
"""sig2hac:

"""

from __future__ import division

import cPickle as pickle

from scikits import audiolab


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print 'Usage: sig2hac.py <hacfile> <fromfile> <tofile>'
    wavfile = sys.argv[2]
    with open(sys.argv[1], 'rb') as f:
        haccer = pickle.load(f)

    sig, fs, _ = audiolab.wavread(wavfile)
    s = haccer.convert_sig(sig)
    with open(sys.argv[3], 'wb') as f:
        pickle.dump(s, f, -1)
