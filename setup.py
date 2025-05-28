from setuptools import setup, find_packages

setup(
    name="montecolor",
    version="0.0",
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'colorspacious', 'colormath', 'emcee', 'seaborn'
    ]
)
