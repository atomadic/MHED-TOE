"""
Tests for revelation tensor.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from mhed_toe.revelation_tensor import RevelationTensor

class TestRevelationTensor(unittest.TestCase):
    def setUp(self):
        self.rt = RevelationTensor()

    def test_tensor_dimensions(self):
        expected_shape = (len(self.rt.iit_axioms), len(self.rt.energy_scales), len(self.rt.physical_regimes))
        self.assertEqual(self.rt.revelation_tensor_data.shape, expected_shape)

    def test_bridge_strength_range(self):
        strength = self.rt.get_bridge_strength(0, 0, 0)
        self.assertIsInstance(strength, float)
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 1.0) # Normalized to be within 0 and 1 (or 0.005 and 0.055 as per init)

    def test_main_output(self):
        result = self.rt.main()
        self.assertIsInstance(result, dict)
        self.assertIn('tensor_shape', result)
        self.assertIn('sample_bridge_strength', result)
        self.assertEqual(result['tensor_shape'], self.rt.revelation_tensor_data.shape)

if __name__ == '__main__':
    unittest.main()
