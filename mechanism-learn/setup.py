from setuptools import setup, find_packages

setup(
    name='mechanism-learn',
    version='1.0.0',
    author='Jianqiao Mao',
    author_email='jxm1417@student.bham.ac.uk',
    license='GPL-3.0',
    description="Mechanism-learn is a simple method which uses front-door causal bootstrapping to deconfound observational data such that any appropriate machine learning model is forced to learn predictive relationships between effects and their causes (reverse causal inference), despite the potential presence of multiple unknown and unmeasured confounding. The library is compatible with most existing ML deployments such as models built with Scikit-learn and Keras.",
    url='https://github.com/JianqiaoMao/CausalBootstrapping',
    py_modules=['mechanismlearn'],
    install_requires=['causalBootstrapping', 'grapl-causal', 'scipy', 'graphviz'],
)