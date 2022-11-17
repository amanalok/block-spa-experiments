from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="block-spa-experiments",
    version="0.2.0",
    description="Block-based Structured Adapter Pruning experiments on BERT-base.",
    long_description=readme(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.0",
        "Topic :: Text Processing",
    ],
    keywords="",
    url="",
    author="",
    author_email="",
    license="Apache 2.0",
    packages=["block_movement_pruning"],
    include_package_data=True,
    zip_safe=False,
)
