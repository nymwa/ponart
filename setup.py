import setuptools

setuptools.setup(
        name = 'ponart',
        packages = setuptools.find_packages(),
        install_requires=[
            'numpy', 'torch', 'tqdm'],)
