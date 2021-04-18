from setuptools import setup, find_packages

setup(
    name='wobert',
    package_dir={"": "src"},
    packages=find_packages("src"),
    version='0.0.1',
    license='MIT',
    description='wobert_pytorch',
    author='Jun Yu',
    author_email='573009727@qq.com',
    url='https://github.com/JunnYu/WoBERT_pytorch',
    keywords=['wobert', 'pytorch'],
    install_requires=['transformers'],
)