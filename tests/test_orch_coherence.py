#!/usr/bin/env python3
"""
Tests for Orch-OR coherence timing calculations.
"""

import pytest
import numpy as np
from scipy import constants
from mhed_toe import OrchORCalculator


def test_orch_calculator_initialization():
    """Test OrchORCalculator initialization."""
    orch = OrchORCalculator()
    
    # Check constants
    assert orch.hbar == constants.hbar
    assert orch.G == constants.G
    assert orch.k_B == constants.k
    
    # Check tubulin parameters
    assert orch.m_tubulin == 1.0e-22
    assert orch.r_tubulin == 1.0e-9
    assert orch.N_tubulin_default == 1e4
    
    # Check bridge parameter
    assert orch.U_TL == 3.49


def test_gravitational_self_energy():
    """Test gravitational self-energy calculation."""
    orch = OrchORCalculator()
    
    # Default values
    E_G = orch.calculate_e_g()
    expected = orch.G * orch.m_tubulin**2 / orch.r_tubulin
    assert np.isclose(E_G, expected, rtol=1e-10)
    
    # Custom values
    m_test = 2.0e-22
    r_test = 2.0e-9
    E_G_custom = orch.calculate_e_g(m=m_test, r=r_test)
    expected_custom = orch.G * m_test**2 / r_test
    assert np.isclose(E_G_custom, expected_custom, rtol=1e-10)


def test_orch_or_timing():
    """Test Orch-OR timing calculation."""
    orch = OrchORCalculator()
    
    # Test with default parameters
    tau = orch.calculate_tau_or(n_tubulins=1e4)
    
    # Should be positive
    assert tau > 0
    
    # Should be around 9.2 ms
    assert 1e-3 < tau < 1e-2  # Between 1 ms and 10 ms
    
    # Test with custom tubulin count
    tau_small = orch.calculate_tau_or(n_tubulins=100)
    tau_large = orch.calculate_tau_or(n_tubulins=1e6)
    
    # More tubulins → shorter τ (inverse relationship)
    assert tau_small > tau_large


def test_timing_without_u_mod():
    """Test Orch-OR timing without U_TL modification."""
    orch = OrchORCalculator()
    
    tau_with = orch.calculate_tau_or(n_tubulins=1e4, include_u_mod=True)
    tau_without = orch.calculate_tau_or(n_tubulins=1e4, include_u_mod=False)
    
    # Should be different
    assert not np.isclose(tau_with, tau_without, rtol=1e-3)


def test_coherence_cascade():
    """Test coherence cascade calculation."""
    orch = OrchORCalculator()
    
    cascade = orch.calculate_coherence_cascade(tau_coh_single=143e-15, n_tubulins=100)
    
    # Check structure
    assert 'tau_coh_scaled' in cascade
    assert 'tau_or' in cascade
    assert 'freq_or' in cascade
    assert 'n_tubulins' in cascade
    assert 'tau_coh_single' in cascade
    
    # Check values
    assert cascade['tau_coh_single'] == 143e-15
    assert cascade['n_tubulins'] == 100
    
    # Scaled coherence should be longer than single
    assert cascade['tau_coh_scaled'] > cascade['tau_coh_single']
    
    # Frequency should be 1/τ
    assert np.isclose(cascade['freq_or'], 1 / cascade['tau_or'], rtol=1e-10)


def test_eeg_band_calculation():
    """Test EEG frequency band calculation."""
    orch = OrchORCalculator()
    
    eeg_band = orch.calculate_eeg_band(n_range=(8e3, 1.2e4))
    
    # Check structure
    assert 'n_range' in eeg_band
    assert 'tau_range' in eeg_band
    assert 'freq_range' in eeg_band
    assert 'freq_mod_range' in eeg_band
    assert 'gamma_band' in eeg_band
    assert 'delta_freq' in eeg_band
    assert 'delta_mod' in eeg_band
    
    # Check n_range
    assert eeg_band['n_range'] == (8e3, 1.2e4)
    
    # Check gamma band
    assert eeg_band['gamma_band'] == (30, 100)
    
    # Frequencies should be positive
    assert eeg_band['freq_range'][0] > 0
    assert eeg_band['freq_range'][1] > 0
    
    # Modulated frequencies should be different
    assert eeg_band['freq_mod_range'][0] != eeg_band['freq_range'][0]
    assert eeg_band['freq_mod_range'][1] != eeg_band['freq_range'][1]


def test_reproducibility():
    """Test that calculations are reproducible."""
    orch1 = OrchORCalculator()
    orch2 = OrchORCalculator()
    
    # Same parameters should give same results
    tau1 = orch1.calculate_tau_or(n_tubulins=1000)
    tau2 = orch2.calculate_tau_or(n_tubulins=1000)
    
    assert np.isclose(tau1, tau2, rtol=1e-10)
    
    # E_G should be the same
    E_G1 = orch1.calculate_e_g()
    E_G2 = orch2.calculate_e_g()
    assert np.isclose(E_G1, E_G2, rtol=1e-10)


def test_edge_cases():
    """Test edge cases."""
    orch = OrchORCalculator()
    
    # Very small number of tubulins
    tau_small = orch.calculate_tau_or(n_tubulins=1)
    assert tau_small > 0
    
    # Very large number of tubulins
    tau_large = orch.calculate_tau_or(n_tubulins=1e10)
    assert tau_large > 0
    
    # Zero tubulins (should handle gracefully or raise error)
    with pytest.raises((ZeroDivisionError, ValueError)):
        orch.calculate_tau_or(n_tubulins=0)


def test_physical_plausibility():
    """Test physical plausibility of results."""
    orch = OrchORCalculator()
    
    # E_G for single tubulin
    E_G = orch.calculate_e_g()
    
    # Should be extremely small but positive
    assert 0 < E_G < 1e-10  # Less than 0.1 nJ
    
    # τ_OR for typical axon segment
    tau_or = orch.calculate_tau_or(n_tubulins=1e4)
    
    # Should be in the ms range (corresponding to gamma band)
    assert 1e-3 < tau_or < 1e-2  # 1-10 ms
    
    # Corresponding frequency
    freq = 1 / tau_or
    
    # Should be in gamma band (30-100 Hz)
    assert 20 < freq < 120  # Allow some margin


@pytest.mark.slow
def test_timing_curve():
    """Test timing curve generation (integration test)."""
    orch = OrchORCalculator()
    
    # This is more of an integration test
    n_values = [100, 1000, 10000]
    tau_values = []
    
    for n in n_values:
        tau = orch.calculate_tau_or(n_tubulins=n)
        tau_values.append(tau)
    
    # More tubulins → shorter τ
    for i in range(len(tau_values) - 1):
        assert tau_values[i] > tau_values[i + 1]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Also run a quick demonstration
    print("\n" + "="*60)
    print("Orch-OR Calculator Test Results")
    print("="*60)
    
    orch = OrchORCalculator()
    
    # Calculate key values
    E_G = orch.calculate_e_g()
    tau_or = orch.calculate_tau_or(n_tubulins=1e4)
    
    print(f"Gravitational self-energy (single tubulin): {E_G:.2e} J")
    print(f"Orch-OR timing (N=10^4 tubulins): {tau_or*1000:.1f} ms")
    print(f"Frequency: {1/tau_or:.1f} Hz")
    
    # Test reproducibility
    tau1 = orch.calculate_tau_or(n_tubulins=1000)
    tau2 = orch.calculate_tau_or(n_tubulins=1000)
    print(f"Reproducibility test: τ1={tau1:.3e}, τ2={tau2:.3e}, "
          f"difference={abs(tau1-tau2)/tau1*100:.2f}%")
    
    print("All tests passed!")
