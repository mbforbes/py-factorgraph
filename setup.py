from setuptools import setup

setup(
    name='factorgraph',
    version='0.0.1',
    author='Maxwell Forbes',
    author_email='mbforbes@cs.uw.edu',
    packages=['factorgraph'],
    url='https://github.com/mbforbes/py-factorgraph/',
    license='MIT',
    description='Factor graph and loopy belief propagation.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.13.0",
    ],
)
