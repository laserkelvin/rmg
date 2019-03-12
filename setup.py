from setuptools import setup

setup(
    name='RMG',
    version='0.2.0',
    packages=['rmg'],
    install_requires=["joblib", "tqdm", "scipy", "networkx", "numpy"],
    url='',
    license='',
    author='Kelvin Lee',
    author_email='kinlee@cfa.harvard.edu',
    description='Random molecule generator'
)
