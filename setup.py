# -*- coding: utf-8 -*-

import re
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def find_version(*file_path_parts):
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, *file_path_parts), "r") as fp:
        version_file_text = fp.read()

    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        version_file_text,
        re.M,
    )
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")


setup(
    name="meta-learning-for-everyone",
    version="latest",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    author="Dongmin Lee, Seunghyun Lee",
    author_email="kid33629@gmail.com, shlee21@postech.ac.kr",
    description="'모두를 위한 메타러닝' 책에 대한 코드 저장소",
    keywords="meta-learning-for-everyone",
    url="https://github.com/dongminlee94/meta-learning-for-everyone",
    project_urls={
        "Documentation": "https://github.com/dongminlee94/meta-learning-for-everyone",
        "Source Code": "https://github.com/dongminlee94/meta-learning-for-everyone",
    },
    install_requires=[],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
