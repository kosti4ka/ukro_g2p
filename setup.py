import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="ukro_g2t_pytorch",
    version="0.1",
    author="Kostiantyn Pylypenko",
    author_email="k.pylypenko@hotmail.com",
    description="NN based grapheme to phoneme model for Ukrainian language",
    license="MIT",
    keywords="Ukrainian grapheme to phoneme",
    url="https://github.com/kosti4ka/ukro_g2p",
    packages=['ukro_g2p'],
    long_description=read('README'),
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
