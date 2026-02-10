"""
Tests for causal set sprinkling.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import networkx as nx
from mhed_toe.causal_set_sprinkling import CausalSetSprinkler

class TestCausalSetSprinkler(unittest.TestCase):
    def setUp(self):
        self.sprinkler = CausalSetSprinkler(rho=5, volume=5) # N=25 for default

    def test_sprinkle_causal_set(self):
        self.sprinkler.sprinkle_causal_set()
        self.assertEqual(len(self.sprinkler.causal_set), self.sprinkler.N)
        # Check if sorted by time
        times = [p[0] for p in self.sprinkler.causal_set]
        self.assertTrue(all(times[i] <= times[i+1] for i in range(len(times) - 1)))

    def test_is_causally_related(self):
        # Future event
        self.assertTrue(self.sprinkler.is_causally_related((0,0), (1,0.5)))
        # Past event
        self.assertFalse(self.sprinkler.is_causally_related((1,0.5), (0,0)))
        # Spacelike event
        self.assertFalse(self.sprinkler.is_causally_related((0,0), (1,2)))
        # Same time (not causally related in this definition)
        self.assertFalse(self.sprinkler.is_causally_related((0,0), (0,0)))

    def test_build_causal_matrix(self):
        self.sprinkler.sprinkle_causal_set()
        self.sprinkler.build_causal_matrix()
        self.assertIsInstance(self.sprinkler.causal_matrix, np.ndarray)
        self.assertEqual(self.sprinkler.causal_matrix.shape, (self.sprinkler.N, self.sprinkler.N))
        # Check if no element causally precedes itself or future precedes past
        self.assertEqual(np.sum(np.diag(self.sprinkler.causal_matrix)), 0) # No self-causation
        self.assertEqual(np.sum(np.tril(self.sprinkler.causal_matrix, k=-1)), 0) # No lower triangle (past -> future only)

    def test_calculate_u_value(self):
        self.sprinkler.sprinkle_causal_set()
        self.sprinkler.build_causal_matrix()
        u_val = self.sprinkler.calculate_u_value()
        self.assertIsInstance(u_val, float)
        self.assertGreaterEqual(u_val, 0.0)

if __name__ == '__main__':
    unittest.main()
