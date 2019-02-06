from io import open

import os, shutil, torch
from setuptools import setup, find_packages
import distutils.command.clean
from torch.utils.cpp_extension import CppExtension

class clean(distutils.command.clean.clean):

    def run(self):
        import glob
        import re
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
            for wildcard in filter(None, ignores.split('\n')):                    

                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    for filename in glob.glob(wildcard):
                        # skip vscode
                        vscode_pat = re.compile(r'.vscode/.*')
                        if re.match(vscode_pat, filename):
                            continue

                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


with open('torch_complex/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRES = []

cmdclass = {
    "build_ext": torch.utils.cpp_extension.BuildExtension,
    'clean': clean,
}

ext_modules = [
    CppExtension(
        "torch_complex.cpp",
        ["src/module.cpp"],
        extra_compile_args=["-g", "-stdlib=libc++", "-std=c++11"],
    )
]

setup(
    name='torch-complex',
    version=version,
    description='',
    long_description=readme,
    author='Roger Luo',
    author_email='rogerluo.rl18@gmail.com',
    maintainer='Roger Luo',
    maintainer_email='rogerluo.rl18@gmail.com',
    url='https://github.com/_/torch-complex',
    license='MIT',

    keywords=[
        '',
    ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],

    install_requires=REQUIRES,
    tests_require=['coverage', 'pytest'],

    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
