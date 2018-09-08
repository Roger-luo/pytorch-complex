# PyTorch-Complex

Complex-valued tensor support for [PyTorch](https://github.com/pytorch/pytorch). (Work in progress)

## Usage

Build this plugin just like a normal Python package:

```sh
python setup.py install
python setup.py build
python setup.py test
```

Due to this issue: https://github.com/pytorch/extension-cpp/issues/6

you need to import `torch` first.

```python
import torch, torch_complex
```

then the complex tensor support will be in `torch` module. Use it just like the other tensor types.
