# This file is part of Rank-based persistence
# Copyright (C) 2019- Mattia G. Bergomi, Pietro Vertechi
#
# Rank-based persistence is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1,2].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please use the tools available at
# [1] Bergomi, M. G., & Vertechi, P. (2019). Rank-based persistence. arXiv preprint arXiv:1905.09151.
# [2] Bergomi, M. G., & Vertechi, P. (2020). Comparing Neural Networks via Generalized Persistence. SIS 2020 - Book of Short Papers.


import os
import sys
from distutils.sysconfig import get_python_lib
from setuptools import find_packages, setup
import subprocess

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 5)
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of rank_persistence requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)

requirements = [
                'Keras>=2.2.4',
                'keras-rl>=0.4.2',
                'matplotlib>=3.0.3',
                'networkx>=2.3',
                'numpy>=1.14.4',
                'scipy>=1.3.0',
                'seaborn>=0.9.0',
                'tensorflow-gpu==1.14'
                ]
EXCLUDE_FROM_PACKAGES = []

setup(
    name='rank_persistence',
    version='0.0.0-prealpha',
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    url='',
    author='',
    author_email='',
    description=(''),
    license='GNU General Public License v3 or later (GPLv3+)',
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    install_requires=requirements,
    entry_points={},
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering'
        'Topic :: Scientific/Engineering :: Machine Learning',
        'Topic :: Scientific/Engineering :: Machine cognition',
    ],
)
