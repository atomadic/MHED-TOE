"""
Tests for Yukawa axon spectrum simulation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from mhed_toe.yukawa_axon_spectrum import YukawaAxonModel

class TestYukawaAxonModel(unittest.TestCase):
    def setUp(self):
        self.model = YukawaAxonModel(N=10)

    def test_generate_fermion_spectrum(self):
        spectrum = self.model.generate_fermion_spectrum()
        self.assertIsInstance(spectrum, dict)
        self.assertIn('u_sim', spectrum)
        self.assertIn('higgs_sim', spectrum)
        self.assertGreater(spectrum['u_sim'], 0)

    def test_validate_spectrum(self):
        self.model.generate_fermion_spectrum()
        errors = self.model.validate_spectrum()
        self.assertIsInstance(errors, dict)
        self.assertIn('average_fermion_error_percent', errors)
        self.assertIn('higgs_error_percent', errors)
        # Check if error is a non-negative float
        self.assertGreaterEqual(errors['average_fermion_error_percent'], 0)

if __name__ == '__main__':
    unittest.main()
