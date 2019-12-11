# coding=utf-8
# @time: 2019/12/11 11:24
# @author: Wang Ke
# @contact: wangke09@58.com
# =========================

from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name="wktk4py",
    version="0.0.1",
    author="Colin Wang",
    author_email="colinwke@gmail.com",
    description="A personal(wk) toolkit for python.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/colinwke/wktk4py",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
