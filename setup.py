from setuptools import setup, find_packages
import os

current_folder = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_folder, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='imagesorter',
    version='1.0.0',
    description='Sort images based on image embeddings similarity',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dashesy/imagesorter',
    author='Ehsan Azar',
    author_email='dashesy@gmail.com',
    license='BSD 2 clause',
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='imagesorter featurizer image similarity',
    packages=find_packages(),
    python_requires='>=3.6'
)