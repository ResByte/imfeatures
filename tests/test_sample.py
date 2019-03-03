# Sample Test passing with unittest

import unittest
from imfeatures import Features
import torch

class TestFeatures(unittest.TestCase):
    def test_feature(self):
        model = Features('resnet18', True)
        x = torch.randn([1,3,224,224])
        with torch.no_grad():
            out = model(x)
        assert(out.squeeze().shape[0] == 512)

if __name__ == "__main__":
    unittest.main()