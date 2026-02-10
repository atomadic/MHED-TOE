"""
Generate all figures for the MHED-TOE paper.
Run this script to recreate all figures from the paper.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mhed_toe import YukawaAxonModel, OrchORCalculator, HolographicPhi
from mhed_toe import CausalSetSprinkler, RevelationTensor

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def create_figures_directory():
    fig_dir = '../figures'
    os.makedirs(fig_dir, exist_ok=True)
    print(f"📁 Figures directory: {fig_dir}")
    return fig_dir

def generate_all_figures():
    print("=" * 60)
    print("Generating MHED-TOE Paper Figures")
    print("=" * 60)
    fig_dir = create_figures_directory()

    # Figure 1: Fermion Spectrum
    print("Generating Figure 1: Fermion Spectrum...")
    yukawa_model = YukawaAxonModel()
    yukawa_model.generate_fermion_spectrum()
    yukawa_model.visualize_spectrum()
    plt.savefig(os.path.join(fig_dir, 'fig1_fermion_spectrum.png'))
    plt.close()

    # Figure 2: Coherence Cascade
    print("Generating Figure 2: Coherence Cascade...")
    orch_calc = OrchORCalculator(N_tubulins=100)
    tau_OR = orch_calc.calculate_orch_or_time()
    gamma_freq = orch_calc.calculate_eeg_gamma_frequency(tau_OR)
    orch_calc.visualize_coherence(tau_OR, gamma_freq)
    plt.savefig(os.path.join(fig_dir, 'fig2_coherence_cascade.png'))
    plt.close()

    # Figure 3: Holographic Phi
    print("Generating Figure 3: Holographic Phi...")
    phi_calc = HolographicPhi(boundary_size=25)
    phi_calc.phi_holo, phi_calc.max_subset = phi_calc.find_max_phi_subset()
    phi_calc.visualize_graph_with_phi()
    plt.savefig(os.path.join(fig_dir, 'fig3_holographic_phi.png'))
    plt.close()

    # Figure 4: Revelation Tensor
    print("Generating Figure 4: Revelation Tensor slices...")
    revelation_tensor = RevelationTensor()
    revelation_tensor.visualize_tensor_slice(fixed_regime_idx=0) # Quantum Gravity
    plt.savefig(os.path.join(fig_dir, 'fig4_revelation_tensor_QG.png'))
    plt.close()
    revelation_tensor.visualize_tensor_slice(fixed_regime_idx=1) # Particle Physics
    plt.savefig(os.path.join(fig_dir, 'fig4_revelation_tensor_PP.png'))
    plt.close()
    revelation_tensor.visualize_tensor_slice(fixed_regime_idx=2) # Neuroscience
    plt.savefig(os.path.join(fig_dir, 'fig4_revelation_tensor_Neuro.png'))
    plt.close()

    # Figure 5: Causal Set Sprinkling (Master Equation visualization is conceptual, not code-generated directly)
    print("Generating Figure 5: Causal Set Sprinkling (for spatial representation of master equation context)...")
    causal_sprinkler = CausalSetSprinkler(rho=10, volume=10)
    causal_sprinkler.sprinkle_causal_set()
    causal_sprinkler.build_causal_matrix()
    causal_sprinkler.visualize_causal_set()
    plt.savefig(os.path.join(fig_dir, 'fig5_causal_set_sprinkling.png'))
    plt.close()

    print(f"
All figures generated and saved to {fig_dir}")

if __name__ == '__main__':
    generate_all_figures()
