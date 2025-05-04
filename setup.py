from setuptools import setup, find_packages

setup(
    name='mechanism-learn',
    version='2.0.2',
    author='Jianqiao Mao',
    author_email='jxm1417@student.bham.ac.uk',
    license='GPL-3.0',
    description="Mechanism-learn is a simple method to deconfound observational data such that any appropriate machine learning model is forced to learn predictive relationships between effects and their causes, despite the potential presence of multiple unknown and unmeasured confounding. The library is compatible with most existing ML deployments. The library is compatible with most existing ML deployments such as models built with Scikit-learn and Keras.",
    url='https://github.com/JianqiaoMao/mechanism-learn',
    packages=find_packages(), 
    install_requires=['causalBootstrapping', 'grapl-causal', 'scipy', 'graphviz', 'tqdm'],
)