import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='wry',
    version='0.1',
    description='Weighted regressions for hydrology.',
    url='https://github.com/thodson-usgs/wry.git',
    author='Timothy Hodson',
    author_email='thodson@usgs.gov',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=setuptools.find_packages(),
    zip_safe=False
)
