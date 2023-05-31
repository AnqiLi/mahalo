from distutils.core import setup

setup(
    name='mahalo',
    version='0.1.0',
    author='Anqi Li',
    author_email='anqil4@cs.washington.edu',
    packages=['mahalo'],
    url='https://github.com/AnqiLi/mahalo',
    license='MIT LICENSE',
    description='Source code for the paper MAHALO: Unifying Offline Reinforcement Learning and Imitation Learning from Observations',
    long_description=open('README.md').read(),
    install_requires=[
        "importlib-metadata==4.13.0",
        "lightATAC@git+https://github.com/chinganc/lightATAC.git@dev"
    ]
)