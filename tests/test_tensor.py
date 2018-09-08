import unittest
from torch_complex import torch

class TestComplexTensor(unittest.TestCase):

    def test_empty(self):
        torch.empty(2, 2, dtype=torch.complex64)
        torch.empty(2, 2, dtype=torch.complex128)

if __name__ == '__main__':
    unittest.main()
