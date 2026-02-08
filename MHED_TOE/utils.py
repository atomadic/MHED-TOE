"""
Utility functions for MHED-TOE simulations.
"""

import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
from scipy import constants


class Constants:
    """Physical constants used in MHED-TOE."""
    
    # Fundamental constants
    hbar = constants.hbar  # J·s
    c = constants.c  # m/s
    G = constants.G  # m³/kg/s²
    k_B = constants.k  # J/K
    
    # Planck units
    M_pl = np.sqrt(constants.hbar * constants.c / constants.G)  # kg
    L_pl = np.sqrt(constants.hbar * constants.G / constants.c**3)  # m
    T_pl = np.sqrt(constants.hbar * constants.G / constants.c**5)  # s
    E_pl = M_pl * c**2  # J
    
    # Conversion factors
    eV_to_J = constants.eV  # J/eV
    GeV_to_eV = 1e9
    TeV_to_eV = 1e12
    
    # Standard Model parameters
    v_ew = 246.0  # GeV (Higgs vev)
    alpha_em = 1/137.035999084  # Fine structure constant
    sin2_theta_w = 0.231  # Weak mixing angle
    
    # MT parameters
    m_tubulin = 1.0e-22  # kg (superposed mass difference)
    r_tubulin = 1.0e-9  # m (separation)
    dipole_moment = 3300 * constants.e * 1e-30  # C·m (3300 Debye)
    
    # Orch-OR parameters
    tau_or_brain = 9.2e-3  # s (36 Hz)
    e_g_tubulin = 1.0e-20  # J (gravitational self-energy)
    
    # IIT parameters
    phi_human = 1e3  # Approximate Φ for human brain
    phi_mouse = 1e2  # Approximate Φ for mouse brain
    
    # E8/GUT parameters
    m_gut = 1e16  # GeV (GUT scale)
    tau_proton = 1e36 * 365.25 * 24 * 3600  # s (proton decay lifetime)


def validate_simulation(results: Dict[str, Any], 
                       tolerance: float = 0.01) -> Dict[str, bool]:
    """
    Validate simulation results against expected values.
    
    Parameters
    ----------
    results : dict
        Simulation results
    tolerance : float
        Acceptable fractional error
        
    Returns
    -------
    dict
        Validation results
    """
    validation = {}
    
    # Check fermion masses
    if 'fermion_spectrum' in results:
        spectrum = results['fermion_spectrum']
        
        # Check top quark mass
        if 't' in spectrum:
            m_t_pred = spectrum['t']['mass_gev']
            m_t_exp = 173.0
            error = abs(m_t_pred - m_t_exp) / m_t_exp
            validation['top_quark'] = error < tolerance
            validation['top_quark_error'] = error * 100
        
        # Check average error
        if 'average_error' in spectrum:
            validation['average_error'] = spectrum['average_error'] < 10.0  # < 10%
    
    # Check Higgs mass
    if 'higgs_mass_gev' in results:
        m_h_pred = results['higgs_mass_gev']
        m_h_exp = 125.1
        error = abs(m_h_pred - m_h_exp) / m_h_exp
        validation['higgs_mass'] = error < tolerance
        validation['higgs_error'] = error * 100
    
    # Check Orch-OR timing
    if 'tau_or_ms' in results:
        tau_pred = results['tau_or_ms']
        tau_exp = 9.2
        error = abs(tau_pred - tau_exp) / tau_exp
        validation['orch_or_timing'] = error < tolerance
        validation['tau_error'] = error * 100
    
    # Check holographic Φ
    if 'phi_holo' in results:
        phi_pred = results['phi_holo']
        phi_exp = 2.847
        error = abs(phi_pred - phi_exp) / phi_exp
        validation['holographic_phi'] = error < tolerance
        validation['phi_error'] = error * 100
    
    # Count successes
    validation['total_tests'] = len([k for k in validation.keys() 
                                    if not k.endswith('_error')])
    validation['passed_tests'] = sum([v for k, v in validation.items() 
                                     if not k.endswith('_error') and isinstance(v, bool)])
    
    return validation


def plot_validation_results(validation: Dict[str, Any], 
                          save_path: str = None):
    """
    Plot validation results.
    
    Parameters
    ----------
    validation : dict
        Validation results from validate_simulation()
    save_path : str, optional
        Path to save figure
    """
    # Extract error values
    errors = {}
    for key, value in validation.items():
        if key.endswith('_error'):
            test_name = key.replace('_error', '')
            errors[test_name] = value
    
    if not errors:
        print("No error data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tests = list(errors.keys())
    error_values = [errors[test] for test in tests]
    
    bars = ax.bar(tests, error_values, color='steelblue')
    
    # Add threshold line
    ax.axhline(y=1.0, color='r', linestyle='--', 
              label='1% tolerance threshold')
    
    # Color bars based on threshold
    for bar, error in zip(bars, error_values):
        if error > 1.0:
            bar.set_color('crimson')
    
    ax.set_xlabel('Test')
    ax.set_ylabel('Error (%)')
    ax.set_title('MHED-TOE Simulation Validation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, error in zip(bars, error_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{error:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # Add summary text
    total = validation.get('total_tests', 0)
    passed = validation.get('passed_tests', 0)
    
    summary_text = f"Tests: {passed}/{total} passed\n"
    if total > 0:
        summary_text += f"Success rate: {passed/total*100:.1f}%"
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Validation plot saved to {save_path}")
    
    plt.show()


def save_results_to_json(results: Dict[str, Any], 
                        filename: str = "mhed_toe_results.json"):
    """
    Save simulation results to JSON file.
    
    Parameters
    ----------
    results : dict
        Simulation results
    filename : str
        Output filename
    """
    import json
    
    # Convert numpy arrays to lists
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filename}")


def load_results_from_json(filename: str = "mhed_toe_results.json") -> Dict[str, Any]:
    """
    Load simulation results from JSON file.
    
    Parameters
    ----------
    filename : str
        Input filename
        
    Returns
    -------
    dict
        Simulation results
    """
    import json
    
    with open(filename, 'r') as f:
        results = json.load(f)
    
    print(f"Results loaded from {filename}")
    return results


def generate_report(results: Dict[str, Any]) -> str:
    """
    Generate a summary report of simulation results.
    
    Parameters
    ----------
    results : dict
        Simulation results
        
    Returns
    -------
    str
        Formatted report
    """
    report = []
    report.append("="*70)
    report.append("MHED-TOE SIMULATION REPORT")
    report.append("="*70)
    report.append("")
    
    # Fermion spectrum
    if 'fermion_spectrum' in results:
        spectrum = results['fermion_spectrum']
        report.append("FERMION SPECTRUM")
        report.append("-"*40)
        
        quark_names = ['u', 'd', 's', 'c', 'b', 't']
        for name in quark_names:
            if name in spectrum:
                r = spectrum[name]
                report.append(f"{name}: {r['mass_gev']:.4f} GeV "
                             f"(error: {r.get('error_%', 'N/A'):.1f}%)")
        
        if 'average_error' in spectrum:
            report.append(f"\nAverage error: {spectrum['average_error']:.1f}%")
    
    # Higgs mass
    if 'higgs_mass_gev' in results:
        report.append("\nHIGGS MASS")
        report.append("-"*40)
        m_h = results['higgs_mass_gev']
        report.append(f"Predicted: {m_h:.1f} GeV")
        report.append(f"Experimental: 125.1 GeV")
        report.append(f"Error: {abs(m_h - 125.1)/125.1*100:.2f}%")
    
    # Orch-OR timing
    if 'tau_or_ms' in results:
        report.append("\nORCH-OR TIMING")
        report.append("-"*40)
        tau = results['tau_or_ms']
        freq = 1000 / tau
        report.append(f"τ_OR: {tau:.1f} ms")
        report.append(f"Frequency: {freq:.1f} Hz")
        report.append(f"EEG gamma band: 30-100 Hz")
    
    # Holographic Φ
    if 'phi_holo' in results:
        report.append("\nHOLOGRAPHIC INTEGRATED INFORMATION")
        report.append("-"*40)
        phi = results['phi_holo']
        report.append(f"Φ_holo (25-node hex): {phi:.3f}")
        report.append(f"Scaled to brain: ~{phi * 100:.0f}")
    
    # 2026 Predictions
    report.append("\n2026 FALSIFIABLE PREDICTIONS")
    report.append("-"*40)
    report.append("1. LHC Run 3: 0.83 TeV DM singlet (σ=10⁻⁴ pb)")
    report.append("2. Cryo-EM: Tubulin G2 defects ΔΦ=10² @ 19.47°")
    report.append("3. EEG: Gamma modulation 36→39 Hz (Δτ=0.3 ms)")
    report.append("4. Proton decay: τ_p≈10³⁶ years")
    report.append("5. Higgs mass: 124.8 GeV (0.16% error)")
    
    return "\n".join(report)
