"""
Orchestrated Objective Reduction (Orch-OR) timing calculations.
Computes τ_OR from microtubule coherence cascade.
"""

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from typing import Tuple, Dict


class OrchORCalculator:
    """
    Calculate Orch-OR timing from microtubule coherence.
    
    Based on Penrose-Hameroff model with MHED-TOE extensions.
    """
    
    # Physical constants
    hbar = constants.hbar  # J·s
    G = constants.G  # m³/kg/s²
    k_B = constants.k  # J/K
    
    def __init__(self):
        # Tubulin parameters
        self.m_tubulin = 1.0e-22  # kg (superposed mass difference)
        self.r_tubulin = 1.0e-9  # m (separation)
        self.N_tubulin_default = 1e4  # Default number of tubulins
        
        # MT parameters
        self.r_mt = 12e-9  # m (microtubule radius)
        
        # Bridge parameters
        self.U_TL = 3.49  # Toroidal Langlands-OR bridge
    
    def calculate_e_g(self, m: float = None, r: float = None) -> float:
        """
        Calculate gravitational self-energy E_G.
        
        Parameters
        ----------
        m : float, optional
            Superposed mass difference (kg)
        r : float, optional
            Separation distance (m)
            
        Returns
        -------
        float
            Gravitational self-energy (J)
        """
        if m is None:
            m = self.m_tubulin
        if r is None:
            r = self.r_tubulin
        
        return self.G * m**2 / r
    
    def calculate_tau_or(self, 
                        n_tubulins: int = None,
                        include_u_mod: bool = True) -> float:
        """
        Calculate Orch-OR timing τ_OR.
        
        Parameters
        ----------
        n_tubulins : int, optional
            Number of coherent tubulins
        include_u_mod : bool
            Include U_TL-OR modification
            
        Returns
        -------
        float
            Orch-OR timing (seconds)
        """
        if n_tubulins is None:
            n_tubulins = self.N_tubulin_default
        
        # Gravitational self-energy for N tubulins
        E_G_single = self.calculate_e_g()
        E_G_total = n_tubulins * E_G_single
        
        # U_TL-OR modification
        if include_u_mod:
            # Area term
            A_MT = np.pi * self.r_mt**2 * n_tubulins / 1e9  # Convert to nm²
            
            # Modified energy
            E_modified = E_G_total + (np.pi * self.U_TL) / (A_MT * 1e-18)  # J
        else:
            E_modified = E_G_total
        
        # OR timing: τ = ħ / E
        tau = self.hbar / E_modified
        
        return tau
    
    def calculate_coherence_cascade(self, 
                                   tau_coh_single: float = 143e-15,
                                   n_tubulins: int = 100) -> Dict:
        """
        Calculate coherence cascade from single tubulin to axon scale.
        
        Parameters
        ----------
        tau_coh_single : float
            Single tubulin coherence time (seconds)
        n_tubulins : int
            Number of tubulins
            
        Returns
        -------
        dict
            Cascade results
        """
        # Superradiant scaling: τ ∝ N^(1/4) / √γ
        # Based on Dicke model and Fröhlich condensates
        tau_scaled = tau_coh_single * (n_tubulins ** 0.25)
        
        # Calculate τ_OR for this scale
        tau_or = self.calculate_tau_or(n_tubulins=n_tubulins)
        
        # Frequency in Hz
        freq_or = 1 / tau_or
        
        return {
            'tau_coh_scaled': tau_scaled,
            'tau_or': tau_or,
            'freq_or': freq_or,
            'n_tubulins': n_tubulins,
            'tau_coh_single': tau_coh_single
        }
    
    def plot_timing_curve(self, n_max: int = 1e6, save_path: str = None):
        """
        Plot τ_OR vs number of tubulins.
        
        Parameters
        ----------
        n_max : int
            Maximum number of tubulins to plot
        save_path : str, optional
            Path to save figure
        """
        n_values = np.logspace(1, np.log10(n_max), 50).astype(int)
        tau_values = []
        freq_values = []
        
        for n in n_values:
            tau = self.calculate_tau_or(n_tubulins=n)
            tau_values.append(tau)
            freq_values.append(1 / tau)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # τ_OR plot
        ax1.loglog(n_values, tau_values, 'b-', linewidth=2)
        ax1.axhline(y=9.2e-3, color='r', linestyle='--', 
                   label='τ_OR = 9.2 ms (36 Hz)')
        ax1.axvline(x=1e4, color='g', linestyle='--',
                   label='N = 10⁴ (typical axon segment)')
        
        ax1.set_xlabel('Number of Tubulins (N)')
        ax1.set_ylabel('τ_OR (seconds)')
        ax1.set_title('Orch-OR Timing vs Tubulin Number')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Frequency plot
        ax2.loglog(n_values, freq_values, 'r-', linewidth=2)
        ax2.axhline(y=36, color='b', linestyle='--', 
                   label='36 Hz (Gamma band)')
        ax2.axhline(y=40, color='b', linestyle=':', 
                   label='40 Hz (Gamma upper)')
        ax2.axvline(x=1e4, color='g', linestyle='--',
                   label='N = 10⁴')
        
        ax2.set_xlabel('Number of Tubulins (N)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title('Orch-OR Frequency vs Tubulin Number')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        # Print key values
        print("\nKey Orch-OR Values:")
        print("-" * 40)
        print(f"Single tubulin E_G: {self.calculate_e_g():.2e} J")
        print(f"τ_OR for N=10⁴: {self.calculate_tau_or(1e4)*1000:.1f} ms")
        print(f"Frequency: {1/self.calculate_tau_or(1e4):.1f} Hz")
    
    def calculate_eeg_band(self, n_range: Tuple[int, int] = (8e3, 1.2e4)) -> Dict:
        """
        Calculate EEG frequency band for given tubulin range.
        
        Parameters
        ----------
        n_range : tuple
            (min, max) tubulin count
            
        Returns
        -------
        dict
            EEG band information
        """
        n_min, n_max = n_range
        
        tau_min = self.calculate_tau_or(n_tubulins=n_max)  # More tubulins → shorter τ
        tau_max = self.calculate_tau_or(n_tubulins=n_min)  # Fewer tubulins → longer τ
        
        freq_min = 1 / tau_max  # Hz
        freq_max = 1 / tau_min  # Hz
        
        # Gamma band modulation from Yukawa defects
        # Δτ ≈ 0.3 ms from yuk=0.1-0.5 variations
        delta_tau = 0.3e-3  # seconds
        
        freq_mod_min = 1 / (tau_min + delta_tau)
        freq_mod_max = 1 / (tau_max + delta_tau)
        
        return {
            'n_range': n_range,
            'tau_range': (tau_min, tau_max),
            'freq_range': (freq_min, freq_max),
            'freq_mod_range': (freq_mod_min, freq_mod_max),
            'gamma_band': (30, 100),  # Hz
            'delta_freq': (freq_max - freq_min),
            'delta_mod': (freq_mod_max - freq_mod_min)
        }


def main():
    """Main demonstration function."""
    print("="*70)
    print("MHED-TOE: ORCHESTRATED OBJECTIVE REDUCTION CALCULATOR")
    print("="*70)
    
    # Create calculator
    orch = OrchORCalculator()
    
    # Calculate key values
    tau_or = orch.calculate_tau_or(n_tubulins=1e4)
    freq_or = 1 / tau_or
    
    print(f"\nOrch-OR Timing (N=10⁴ tubulins):")
    print(f"  τ_OR = {tau_or*1000:.1f} ms")
    print(f"  Frequency = {freq_or:.1f} Hz")
    
    # Coherence cascade
    cascade = orch.calculate_coherence_cascade(tau_coh_single=143e-15, n_tubulins=100)
    print(f"\nCoherence Cascade (N=100):")
    print(f"  τ_coh (single) = {cascade['tau_coh_single']*1e15:.1f} fs")
    print(f"  τ_coh (scaled) = {cascade['tau_coh_scaled']*1e15:.1f} fs")
    print(f"  τ_OR = {cascade['tau_or']*1000:.1f} ms")
    print(f"  Frequency = {cascade['freq_or']:.1f} Hz")
    
    # EEG band calculation
    eeg_band = orch.calculate_eeg_band(n_range=(8e3, 1.2e4))
    print(f"\nEEG Gamma Band Prediction:")
    print(f"  Frequency range: {eeg_band['freq_range'][0]:.1f}-{eeg_band['freq_range'][1]:.1f} Hz")
    print(f"  Modulated range: {eeg_band['freq_mod_range'][0]:.1f}-{eeg_band['freq_mod_range'][1]:.1f} Hz")
    print(f"  Gamma band: {eeg_band['gamma_band'][0]}-{eeg_band['gamma_band'][1]} Hz")
    
    # Plot timing curve
    orch.plot_timing_curve(n_max=1e6, save_path="figures/orch_or_timing.png")
    
    return orch


if __name__ == "__main__":
    orch = main()
