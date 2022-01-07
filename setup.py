import setuptools

with open("README.md", "r",encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="abcmrt16",
    author="NTIA, PSCR",
    author_email="PSCR@PSCR.gov",
    description="Package to run ABC-MRT16 intelligibility tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usnistgov/abcmrt16",
    packages=setuptools.find_packages(exclude=("tests",)),
    include_package_data=True,
    package_data={'':['templates','ABC_MRT_FB_templates.mat']},
    use_scm_version={'write_to' : 'abcmrt/version.py'},
    setup_requires=['setuptools_scm'],
    license='NIST software License',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Public Domain",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy','scipy'
    ],
    python_requires='>=3.6',
)
