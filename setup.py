#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


setup(
    name='cnntext',
    version='0.1.1',
    description='Denny Britz Yoon Kim CNN in Tensorflow.',
    author='Todd Young',
    author_email='youngmt1@ornl.gov',
    url='https://github.com/yngtodd/hammer',
    packages=[
        'cnntext',
    ],
    package_dir={'cnntext': 'cnntext'},
    include_package_data=True,
    install_requires=[
        'toml'
    ],
    license='MIT',
    zip_safe=False,
    keywords='cnntext',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
