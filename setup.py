import setuptools

try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except Exception:
    long_description = ""

setuptools.setup(
    name="name_recommender",
    version="0.1",
    author="Lasse Valentini Jensen",
    author_email="",
    description="Name recommender POC.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lassevalentini/NameRecommender",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
