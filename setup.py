import setuptools
import itertools

with open("README.md", "r") as fh:
    long_description = fh.read()


__version__ = '0.0.1a5'

INSTALL_REQUIRES = ["numpy>=1.16.2",
                    "scipy>=1.0",
                    "tqdm>=4.29.0",
                    "h5py>=2.5.0",
                    "dipy>=1.0.0",
                    "lpqtree>=0.0.4"],

TESTS_REQUIRE = ['pytest',
                 'pytest-cov']

# Extra requirements, add a keyword 'all' with all extra dependencies
EXTRAS_REQUIRE = {'numba': ['numba>=0.53'],
                  'fury': ['fury>=0.6']}
EXTRAS_REQUIRE['all'] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))


setuptools.setup(
    name='tractosearch',
    version=__version__,
    author='Etienne St-Onge',
    author_email='Firstname.Lastname@usherbrooke.ca',
    url='https://github.com/StongeEtienne/tractosearch',
    description='Fast Tractography Streamline Search',
    long_description='',
    license='BSD 2-Clause',
    packages=['tractosearch'],
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    extras_require=EXTRAS_REQUIRE,
    zip_safe=False,
)
