# TODO: update NODE fw to current version (using version from 1/2020)

import setuptools

setuptools.setup(
    name="torchdiffeq",
    version="0.0.1",
    author="Ricky Tian Qi Chen",
    author_email="rtqichen@cs.toronto.edu",
    description="ODE solvers and adjoint sensitivity analysis in PyTorch.",
    url="https://github.com/rtqichen/torchdiffeq",
    packages=['torchdiffeq', 'torchdiffeq._impl'],
    install_requires=['torch>=0.4.1'],
    classifiers=(
        "Programming Language :: Python :: 3"),)
