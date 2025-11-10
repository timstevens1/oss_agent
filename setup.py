from setuptools import setup, find_packages

setup(
    name="oss_agent",
    version="0.1.0",
    description="gpt_oss_20b powered agent designed to run on apple silicon",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.10,<3.15",
    install_requires=[],
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: OS Independent",
    ],
)