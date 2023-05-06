from setuptools import setup

setup(
   name='activity_prediction',
   version='0.1.0',
   author='Rafal Bachorz',
   author_email='rafal@bachorz.eu',
   packages=['auxiliary', 'constants', 'data_transformers', 'model_wrapper'],
   #scripts=['bin/script1','bin/script2'],
   url='http://pypi.python.org/pypi/PackageName/',
   license='MIT',
   description='An awesome package that does something',
   long_description=open('README').read(),
   #install_requires=[
   #    "Django >= 1.1.1",
   #    "pytest",
   #],
)