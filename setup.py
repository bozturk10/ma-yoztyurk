from setuptools import find_packages
from setuptools import setup

setup(
    name='src',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    license='MIT',
    author='Your NAME',
    author_email='your@email.com',
    description='Your main project'
)