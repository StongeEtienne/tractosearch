import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


INSTALL_REQUIRES = ["numpy>=1.16.2",
                    "scipy>=1.0",
                    "h5py>=2.5.0",
                    "vtk>=8.1.2",
                    "tqdm>=4.29.0",
                    "dipy>=1.0.0"],

TESTS_REQUIRE = ['pytest', 'pytest-cov']

setuptools.setup(
    name='tractosearch',
    version=__version__,
    author='Etienne St-Onge',
    author_email='Firstname.Lastname@usherbrooke.ca',
    url='https://github.com/StongeEtienne/tractosearch',
    description='Fast Tractography Streamline Search',
    long_description='',
    license='BSD 3-Clause',
    ext_modules=ext_modules,
    packages=['tractosearch'],
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    zip_safe=False,
)
