from setuptools import setup

setup(
    name="factorgraph",
    version="0.0.3",
    author="Maxwell Forbes",
    author_email="mbforbes@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="factorgraph factor graph loopy belief propagation lbp sum product",
    packages=["factorgraph"],
    url="https://github.com/mbforbes/py-factorgraph/",
    license="MIT",
    description="Factor graphs and loopy belief propagation library.",
    install_requires=[
        "numpy >= 1.13.0",
    ],
)
