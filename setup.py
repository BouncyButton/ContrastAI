#! /usr/bin/env python
"""A template for scikit-learn compatible packages."""

import codecs
import os

from setuptools import find_packages, setup
import numpy

DISTNAME = 'contrast_ai'
VERSION = '0.0.1'
DESCRIPTION = 'Experimentation for using generative models applied to mammography.'
# with codecs.open('README.rst', encoding='utf-8-sig') as f:
#    LONG_DESCRIPTION = f.read()
MAINTAINER = 'L. Bergamin'
MAINTAINER_EMAIL = 'todo'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/BouncyButton/ContrastAI'
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn', 'pandas']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']

setup(name=DISTNAME,
      version=VERSION,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      download_url=DOWNLOAD_URL,
      # long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      include_dirs=[numpy.get_include()])
