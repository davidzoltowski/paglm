#!/usr/bin/env python

from distutils.core import setup

setup(name='paglm',
      version='0.1.0',
      description='Poisson PASS GLM in python',
      author='David Zoltowski',
      url='https://github.com/davidzoltowski/paglm',
      packages=['paglm'],
      install_requires=['numpy','scipy','matplotlib']
      )
