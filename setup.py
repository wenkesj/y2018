# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
  name='y2018',
  version='1.0.0',
  long_description='Thought experiments and paper implementations in 2018 covering ML',
  author='Sam Wenke',
  author_email='samwenke@gmail.com',
  license='MIT',
  description='Thought experiments and paper implementations in 2018 covering ML',
  packages=find_packages('.'),
  install_requires=['observations', 'numpy', 'tensorflow'],
  platforms='any',
)
