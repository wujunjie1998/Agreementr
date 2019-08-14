import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="agreementr",
    version="1.3",
    author="Junjie Wu",
    author_email="wujj38@mail2.sysu.edu.cn",
    description="agreementr",
    python_requires=">=3.6",
    include_package_data=True,
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    #package_data = {'':['data']}
)