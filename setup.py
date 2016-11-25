from setuptools import setup, find_packages

setup(
    name='lightweight-wavenet',
    version='1.0.0',
    description='Lightweight WaveNet Framework',
    author='Philippe Remy',
    license='MIT',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['tensorflow>=0.10']
)
