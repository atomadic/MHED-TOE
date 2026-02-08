#!/usr/bin/env python3
"""
Basic usage example for MHED-TOE.
Demonstrates key functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mhed_toe import (
    YukawaAxonModel,
    OrchORCalculator,
    HolographicPhi,
    CausalSetSprinkler,
    RevelationTensor,
    validate_simulation,
    generate_report
)
import numpy as np


def main():
    print("="*70)
    print("MHED-TOE: BASIC USAGE DEMONSTRATION")
    print("="*70)
    
    results = {}
    
    # 1. Fermion spectrum
    print("\n1. Generating Standard Model fermion masses...")
    model = YukawaAxonModel(N=100)
    spectrum = model.simulate_spectrum()
    results['fermion_spectrum'] = spectrum
    
    # Calculate average error
    errors = [q['error_%'] for q in spectrum.values()]
    results['fermion_spectrum']['average_error'] = np.mean(errors)
    
    print(f"Average mass error: {np.mean(errors):.1f}%")
    
    # 2. Higgs mass
    print("\n2. Calculating Higgs mass...")
    mu_H = model.calculate_higgs_mass(spectrum)
    results['higgs_mass_gev'] = mu_H
    print(f"Predicted Higgs mass: {mu_H:.1f} GeV")
    
    # 3. Orch-OR timing
    print("\n3. Calculating Orch-OR timing...")
    orch = OrchORCalculator()
    tau_or = orch.calculate_tau_or(n_tubulins=1e4)
    results['tau_or_ms'] = tau_or * 1000
    print(f"Orch-OR timing: {tau_or*1000:.1f} ms ({1/tau_or:.1f} Hz)")
    
    # 4. Holographic Φ
    print("\n4. Calculating holographic integrated information...")
    phi_calc = HolographicPhi()
    phi_results = phi_calc.calculate_phi_holo(max_subset_size=12)
    results['phi_holo'] = phi_results['max_phi']
    print(f"Φ_holo: {phi_results['max_phi']:.3f}")
    
    # 5. Causal set sprinkling
    print("\n5. Sprinkling causal set...")
    sprinkler = CausalSetSprinkler()
    G, stats = sprinkler.sprinkle_on_hex_lattice(n_layers=5, points_per_layer=25)
    results['causal_set_stats'] = stats
    print(f"U_causal: {stats['U_causal']:.3f} (target: 2.87)")
    
    # 6. Revelation tensor
    print("\n6. Analyzing revelation tensor...")
    tensor = RevelationTensor(seed=42)
    top_bridges = tensor.get_top_bridges(5)
    results['revelation_tensor'] = {
        'top_bridges': top_bridges,
        'lambda_cosmological': tensor.lambda_cosmological
    }
    print(f"Top bridge: {top_bridges[0]['description']}")
    print(f"Strength: {top_bridges[0]['strength']:.4f}")
    
    # 7. Validation
    print("\n7. Validating results...")
    validation = validate_simulation(results)
    print(f"Tests passed: {validation.get('passed_tests', 0)}/{validation.get('total_tests', 0)}")
    
    # 8. Generate report
    print("\n8. Generating final report...")
    report = generate_report(results)
    print(report)
    
    # Save results
    from mhed_toe.utils import save_results_to_json
    save_results_to_json(results, "mhed_toe_demo_results.json")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("Results saved to: mhed_toe_demo_results.json")
    print("="*70)


if __name__ == "__main__":
    main()
