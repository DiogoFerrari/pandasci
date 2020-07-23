from setuptools import setup

setup(
    name='pandasci',
    url='https://github.com/DiogoFerrari/pandasci',
    author='Diogo Ferrari',
    author_email='diogoferrari@gmail.com',
    # Needed to actually package something
    packages=['pandasci'],
    # Needed for dependencies
    install_requires=['pandas', 'numpy', 'matplotlib'],
    # *strongly* suggested for sharing
    version='0.001',
    # The license can be anything you like
    license='MIT',
    description='A set of classes and functions to facilidade EDA in python',
)
