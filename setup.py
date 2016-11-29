from setuptools import setup, find_packages

setup(
    name='wavenet',
    version='2.0.0',
    description='Lightweight WaveNet Framework',
    author='Philippe Remy',
    license='MIT',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['tensorflow>=0.10']
)
