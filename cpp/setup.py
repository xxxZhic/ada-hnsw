"""Build the b-only extension on the Linux experiment host.

From this directory: python setup.py build_ext --inplace
Requires a C++17 compiler, setuptools and pybind11.
Set HNSWLIB_NO_NATIVE=1 for a portable, non-march=native build.
"""
import os
from glob import glob

import pybind11
from setuptools import Extension, setup

flags = ["-O3", "-fopenmp", "-std=c++17", "-fvisibility=hidden"]
if not os.environ.get("HNSWLIB_NO_NATIVE"):
    flags.append("-march=native")

setup(
    name="ada-construction",
    version="2.0.0",
    description="Adaptive HNSW construction using normalized frontier distances",
    python_requires=">=3.10",
    ext_modules=[Extension(
        "ada_hnswlib",
        ["bindings.cpp"],
        include_dirs=[pybind11.get_include(), "."],
        depends=sorted(glob("hnswlib/*.h") + ["b_only_policy.h"]),
        language="c++",
        extra_compile_args=flags,
        extra_link_args=["-fopenmp", "-pthread"],
    )],
    install_requires=["numpy>=1.26"],
    zip_safe=False,
)
