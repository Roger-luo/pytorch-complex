import unittest
from torch_complex import torch

class TestComplexTensor(unittest.TestCase):

    def test_empty(self):
        torch.empty(2, 2, dtype=torch.complex64)
        torch.empty(2, 2, dtype=torch.complex128)

    def test_indexing(self):
        t = torch.empty(2, 2, dtype=torch.complex128)
        t[1]
        t[1, 1]

    def test_fill(self):
        t = torch.empty(2, 2, dtype=torch.complex128)
        t.fill_(1.0)
        t.fill_(1.0 + 2.0j)
    
    def test_scalar_binary_op(self):
        a = torch.ones(2, 2, dtype=torch.complex128)
        2 * a
        2 / a
        2 - a
        2 + a



if __name__ == '__main__':
    unittest.main()
