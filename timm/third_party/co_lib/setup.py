import setuptools
# from co_lib import __version__


def read_requirements():
    reqs = []
    with open('requirements.txt', 'r') as fin:
        for line in fin.readlines():
            reqs.append(line.strip())
    return reqs


setuptools.setup(
    name="co_lib",
    version='0.01',
    author="Guanhua Ding",
    author_email="",
    description=("Compression library"),
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    classifiers=(
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3"),
    install_requires=read_requirements()
)
