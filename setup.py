# -*- coding: utf-8 -*-
# flake8: noqa

"""Installation script."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import re

from setuptools import setup


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

def _package_tree(pkgroot):
    path = op.dirname(__file__)
    subdirs = [op.relpath(i[0], path).replace(op.sep, '.')
               for i in os.walk(op.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return subdirs


curdir = op.dirname(op.realpath(__file__))
readme = open(op.join(curdir, 'README.md')).read()


# Find version number from `__init__.py` without executing it.
filename = op.join(curdir, 'klusta/__init__.py')
with open(filename, 'r') as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)


setup(
    name='klusta',
    version=version,
    license="BSD",
    description='Spike detection and automatic clustering for spike sorting',
    long_description=readme,
    author='Kwik Team',
    author_email='cyrille.rossant at gmail.com',
    url='https://klusta.cortexlab.net',
    packages=_package_tree('klusta'),
    package_dir={'klusta': 'klusta'},
    package_data={
        'klusta': ['*.txt', '*.prb'],
    },
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'klusta = klusta.launch:main'
        ],
    },
    include_package_data=True,
    keywords='klusta,neuroscience,spike sorting,klustakwik',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
