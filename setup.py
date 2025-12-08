# Copyright (C) 2022-2023, Henry Gilbert, Ruida Zeng, Michael Sandborn, Jules White, Douglas C. Schmidt
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from setuptools import setup

setup(
    name='psig-matcher',
    version='0.1',
    packages=['psig_matcher'],
    license='GPL 3',
    author='Henry Gilbert, Ruida Zeng, Michael Sandborn, Jules White, Douglas C. Schmidt',
    author_email='henry.gilbert@vanderbilt.edu',
    description="Counterfeit detection via piezoelectric impedance signature analysis",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/matcher',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
    ],
    keywords='piezoelectric, counterfeit detection, impedance signature, supply chain security',
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "perlin_noise",
        "plotly",
    ],
    extras_require={
        'experiments': ['mlflow'],
    },
    entry_points={
        'console_scripts': [
            'psig-matcher = psig_matcher.__main__:run'
        ]
    }
)