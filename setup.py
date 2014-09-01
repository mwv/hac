#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


readme = open('README.org').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = [
    spectral
]

test_requirements = [
    spectral
]

setup(
    name=hac',
    version='0.1.0',
    description='HAC implementation',
    long_description=readme + '\n\n' + history,
    author='Maarten Versteegh',
    author_email='maartenversteegh@gmail.com',
    url='https://github.com/mwv/hac',
    packages=[
        hac',
    ],
    package_dir={'hac':
                 'hac'},
    include_package_data=True,
    install_requires=requirements,
    license="GPL3",
    zip_safe=False,
    keywords='HAC',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GPLv3 License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
