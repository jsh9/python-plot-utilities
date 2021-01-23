from setuptools import setup

setup(
    name='plot_utils',
    version='v0.6.7',
    description='A Python library for elegant data visualization',
    author='Jian Shi',
    license='GPL v3.0',
    url='https://github.com/jsh9/python-plot-utilities',
    packages = ['plot_utils'],
    classifiers = [
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=[
        'numpy>=1.11.0',
        'scipy>=0.19.0',
        'pandas>=0.17.1',
        'cycler>=0.10.0',
        'matplotlib',
    ],
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
    include_package_data=True,
)
