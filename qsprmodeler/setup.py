from setuptools import setup

setup(
   name='qsprmodeler',
   version='0.1.0',
   author='Rafal A. Bachorz',
   author_email='rafal@bachorz.eu',
   packages=['analyzers', 'auxiliary', 'constants', 'data_transformers', 'model_wrapper'],
   package_data = {'data_transformers': ['molecular_descriptors_feature_names_1.2.0.json', 'molecular_meta_descriptors_feature_names_1.2.0.json']},
   license='MIT',
   description='An awesome package that does something',
   #long_description=open('README').read(),
   install_requires=[
      'urllib3 < 2.0',
      'numpy <= 1.24.3',
      'dill >= 0.3.7',
      'prometheus_client',
      'networkx == 2.*',
      'pandas < 2.0',
      'rdkit',
      'scikit-learn >= 1.3.0',
      'mordred == 1.2.0',
      'xgboost >= 1.7.6',
      'hyperopt >= 0.2.7',
      'tensorflow >= 2.13.0',
      'matplotlib',
      "pytest",
   ],
)