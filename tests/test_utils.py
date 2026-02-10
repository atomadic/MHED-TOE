"""
Tests for utility functions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import json
import tempfile
from mhed_toe.utils import Constants, validate_simulation, save_results, load_results, calculate_error, monte_carlo_error_propagation, latex_table, generate_summary_stats

class TestConstants(unittest.TestCase):
    def setUp(self):
        self.const = Constants()

    def test_constants_values(self):
        self.assertAlmostEqual(self.const.hbar, 1.054571817e-34)
        self.assertAlmostEqual(self.const.G, 6.67430e-11)

class TestValidateSimulation(unittest.TestCase):
    def test_all_passed(self):
        results = {"key1": 1.0, "key2": 2.0}
        expected = {"key1": 1.01, "key2": 2.02}
        self.assertTrue(validate_simulation(results, expected, tolerance=0.02)) # 1% error is within 2% tolerance

    def test_some_failed(self):
        results = {"key1": 1.0, "key2": 2.0}
        expected = {"key1": 1.05, "key2": 2.01}
        self.assertFalse(validate_simulation(results, expected, tolerance=0.01)) # 5% error is not within 1% tolerance

    def test_no_expected_values(self):
        results = {"key1": 1.0}
        self.assertTrue(validate_simulation(results, None))

class TestSaveLoadResults(unittest.TestCase):
    def test_save_and_load(self):
        test_data = {"val1": 1.23, "val2": "test", "array_val": np.array([1, 2, 3])}
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_results.json")
            save_results(test_data, filepath)
            loaded_data = load_results(filepath)
            self.assertEqual(loaded_data['val1'], test_data['val1'])
            self.assertEqual(loaded_data['val2'], test_data['val2'])
            self.assertEqual(loaded_data['array_val'], test_data['array_val'].tolist())

class TestCalculateError(unittest.TestCase):
    def test_zero_error(self):
        self.assertEqual(calculate_error(10.0, 10.0), 0.0)

    def test_positive_error(self):
        self.assertAlmostEqual(calculate_error(10.0, 11.0), 10.0)

    def test_infinite_error(self):
        self.assertEqual(calculate_error(0.0, 5.0), float('inf'))

class TestMonteCarloErrorPropagation(unittest.TestCase):
    def test_basic_propagation(self):
        def func(x, y): return x + y
        params = {"x": 10.0, "y": 5.0}
        uncertainties = {"x": 1.0, "y": 0.5}
        mean, std = monte_carlo_error_propagation(func, params, uncertainties, n_samples=1000)
        self.assertAlmostEqual(mean, 15.0, delta=0.5) # Allow some delta for MC simulation
        self.assertAlmostEqual(std, np.sqrt(1.0**2 + 0.5**2), delta=0.5)

class TestLatexTable(unittest.TestCase):
    def test_table_format(self):
        data = [["Row1Col1", 1.23], ["Row2Col1", 4.56]]
        headers = ["Header1", "Header2"]
        table = latex_table(data, headers, caption="Test Table", label="tab:test")
        self.assertIn("\begin{table}[ht]", table)
        self.assertIn("\caption{Test Table}", table)
        self.assertIn("\begin{tabular}{@{}ll{}}", table) # Corrected format for ll
        self.assertIn("Row1Col1 & 1.23 \\n", table) # Corrected escaping

class TestGenerateSummaryStats(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(generate_summary_stats([]), {})

    def test_single_run(self):
        results = [{ "val1": 10, "val2": 20.0 }]
        summary = generate_summary_stats(results)
        self.assertAlmostEqual(summary['val1_mean'], 10.0)
        self.assertAlmostEqual(summary['val2_mean'], 20.0)
        self.assertAlmostEqual(summary['val1_std'], 0.0)

    def test_multiple_runs(self):
        results = [
            { "val1": 10, "val2": 20.0 },
            { "val1": 12, "val2": 22.0 },
            { "val1": 8, "val2": 18.0 }
        ]
        summary = generate_summary_stats(results)
        self.assertAlmostEqual(summary['val1_mean'], 10.0)
        self.assertAlmostEqual(summary['val1_std'], np.std([10, 12, 8]))
        self.assertAlmostEqual(summary['val2_mean'], 20.0)

if __name__ == '__main__':
    unittest.main()
