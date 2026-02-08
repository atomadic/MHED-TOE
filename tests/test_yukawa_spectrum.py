"""
Tests for Yukawa axon spectrum simulation.
"""

import pytest
import numpy as np
from mhed_toe import YukawaAxonModel


def test_model_initialization():
    """Test model initialization."""
    model = YukawaAxonModel(N=100)
    assert model.N == 100
    assert model.j == 50
    assert model.U_TL == 3.49
    assert model.J == 0.1
    assert model.B == 0.01


def test_hamiltonian_construction():
    """Test Hamiltonian construction."""
    model = YukawaAxonModel(N=100)
    H = model.construct_hamiltonian(yuk=0.1)
    
    # Check Hamiltonian properties
    assert H.isherm  # Should be Hermitian
    assert H.dims[0][0] == 101  # Spin dimension (2j+1)
    assert H.dims[1][0] == 10   # Boson dimension


def test_single_generation_simulation():
    """Test single generation simulation."""
    model = YukawaAxonModel(N=50)  # Smaller for speed
    results = model.simulate_generation(yuk=0.1, t_max=10, n_points=20)
    
    # Check results structure
    assert 'yuk' in results
    assert 'delta_m_ev' in results
    assert 'mass_gev' in results
    assert 'tau_coh_fs' in results
    
    # Check value ranges
    assert 0.0 < results['delta_m_ev'] < 0.1  # eV range
    assert 0.0 < results['mass_gev'] < 10.0   # GeV range
    assert 0.0 < results['tau_coh_fs'] < 1000  # fs range


def test_spectrum_simulation():
    """Test full spectrum simulation."""
    model = YukawaAxonModel(N=50)  # Smaller for speed
    results = model.simulate_spectrum(yukawas=[0.1, 0.15])
    
    # Check structure
    assert 'u' in results
    assert 'd' in results
    
    # Check errors calculated
    assert 'error_%' in results['u']
    assert 'error_%' in results['d']


def test_higgs_mass_calculation():
    """Test Higgs mass calculation."""
    model = YukawaAxonModel(N=100)
    results = model.simulate_spectrum(yukawas=[0.1, 0.15])
    mu_H = model.calculate_higgs_mass(results)
    
    # Higgs mass should be around 125 GeV
    assert 100.0 < mu_H < 150.0


@pytest.mark.slow
def test_reproducibility():
    """Test that results are reproducible."""
    model1 = YukawaAxonModel(N=50)
    results1 = model1.simulate_generation(yuk=0.1, t_max=5, n_points=10)
    
    model2 = YukawaAxonModel(N=50)
    results2 = model2.simulate_generation(yuk=0.1, t_max=5, n_points=10)
    
    # Masses should be similar (within 1%)
    np.testing.assert_allclose(results1['mass_gev'], results2['mass_gev'], 
                              rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
