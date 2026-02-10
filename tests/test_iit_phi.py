"""
Tests for holographic IIT calculations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import networkx as nx
from mhed_toe.iit_holographic_phi import HolographicPhi

class TestHolographicPhi(unittest.TestCase):
    def setUp(self):
        self.phi_calc = HolographicPhi(boundary_size=10) # Using a smaller size for faster tests

    def test_create_hex_lattice(self):
        G = self.phi_calc.create_hex_lattice()
        self.assertIsInstance(G, nx.Graph)
        self.assertGreater(G.number_of_nodes(), 0)

    def test_calculate_integrated_information(self):
        # Test with a single node
        nodes = list(self.phi_calc.G.nodes())
        if nodes:
            phi_val = self.phi_calc.calculate_integrated_information({nodes[0]})
            self.assertIsInstance(phi_val, float)
            self.assertGreaterEqual(phi_val, 0.0)

        # Test with an empty set
        phi_val_empty = self.phi_calc.calculate_integrated_information(set())
        self.assertEqual(phi_val_empty, 0.0)

    def test_find_max_phi_subset(self):
        max_phi, best_subset = self.phi_calc.find_max_phi_subset()
        self.assertIsInstance(max_phi, float)
        self.assertGreaterEqual(max_phi, 0.0)
        self.assertIsInstance(best_subset, set)
        # For boundary_size=10, there should be some subset found
        self.assertGreater(len(best_subset), 0)

if __name__ == '__main__':
    unittest.main()
