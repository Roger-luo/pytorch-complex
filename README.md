# PyTorch-Complex

Complex-valued tensor support for [PyTorch](https://github.com/pytorch/pytorch). (Work in progress)

## Usage

**Warning**: this package may require a fresh new build of the PyTorch master branch.

Build this plugin just like a normal Python package:

```sh
python setup.py install
python setup.py build
python setup.py test
```

```python
from torch_complex import torch
```

or

```python
import torch_complex.torch as torch
```

then the complex tensor support will be in `torch` module. Use it just like the other tensor types.
