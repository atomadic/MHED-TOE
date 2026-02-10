"""
Tests for Orch-OR coherence calculations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from mhed_toe.orch_mt_coherence import OrchORCalculator

class TestOrchORCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = OrchORCalculator(N_tubulins=10)

    def test_calculate_gravitational_self_energy(self):
        delta_E_G = self.calc.calculate_gravitational_self_energy()
        self.assertIsInstance(delta_E_G, float)
        self.assertGreater(delta_E_G, 0)

    def test_calculate_orch_or_time(self):
        tau_OR = self.calc.calculate_orch_or_time()
        self.assertIsInstance(tau_OR, float)
        self.assertGreater(tau_OR, 0)
        # A rough check for the expected order of magnitude (~ms)
        self.assertLess(tau_OR, 1.0) # Should be much less than 1 second
        self.assertGreater(tau_OR, 1e-6) # Should be greater than microsecond

    def test_calculate_eeg_gamma_frequency(self):
        tau_OR = self.calc.calculate_orch_or_time()
        gamma_freq = self.calc.calculate_eeg_gamma_frequency(tau_OR)
        self.assertIsInstance(gamma_freq, float)
        self.assertGreater(gamma_freq, 0)
        # Expected gamma freq is around 30-100 Hz
        self.assertGreater(gamma_freq, 10)
        self.assertLess(gamma_freq, 1000)

if __name__ == '__main__':
    unittest.main()
