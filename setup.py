import os
from setuptools import setup, find_packages
from os import path


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="ukro_g2p",
    version="0.1",
    author="Kostiantyn Pylypenko",
    author_email="k.pylypenko@hotmail.com",
    description="NN based grapheme to phoneme model for Ukrainian language",
    license="MIT",
    keywords="Ukrainian grapheme to phoneme",
    url="https://github.com/kosti4ka/ukro_g2p",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
