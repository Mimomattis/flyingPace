from setuptools import setup, find_packages

setup(name='flyingpace',
    version='0.1',
    author='Mattis Gossler',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'fabric',
        'patchwork',
        'ase',
        'numpy',
        'pandas'
    ],
    scripts=['bin/flyingPace'],
    package_data={
        'flyingpace': ['templates/*'],
    },
    )
