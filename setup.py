import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='libmod',
    version='0.1',
    description='Lasso-Impute-Bootstrap Model.',
    url='https://github.com/thodson-usgs/libmod.git',
    author='Timothy Hodson',
    author_email='thodson@usgs.gov',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='CC0',
    packages=setuptools.find_packages(),
    zip_safe=False
)
