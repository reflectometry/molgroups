from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='molgroups',
    version='0.2.0a0',
    packages=['molgroups'],
    url='https://github.com/criosx/molgroups',
    license='MIT License',
    author='Frank Heinrich, David Hoogerheide, Alyssa Thomas',
    author_email='mail@frank-heinrich.net',
    description='Molecular Modeling for Scattering Data Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "bumps", "refl1d", "periodictable", "scipy", "numpy", "matplotlib", "pandas", "scikit-learn",
        "sasmodels", "sasdata", "dill",
    ]
)
