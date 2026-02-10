"""
Run all MHED-TOE simulations and generate complete results.
This script reproduces all key results from the paper.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mhed_toe import YukawaAxonModel, OrchORCalculator, HolographicPhi
from mhed_toe import CausalSetSprinkler, RevelationTensor, IntegratedMHED
from mhed_toe.utils import Constants, validate_simulation, save_results

def main():
    print("=" * 70)
    print("MHED-TOE: Complete Simulation Suite")
    print("=" * 70)
    print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    integrated_mhed = IntegratedMHED()
    all_sim_results, validation_status = integrated_mhed.run_all_simulations()

    # Define expected values for the final validation against paper results
    final_expected_values = {
        "yukawa_model.fermion_masses.higgs_sim": 124.8, # GeV (accessing specific nested key)
        "orch_calculator.gamma_freq": 36.0, # Hz
        "holographic_phi.phi_holo": 2.847,
        "causal_sprinkler.u_hex_value": 2.84
    }

    # Flatten results for final comprehensive validation display
    flat_results_for_final_validation = {}
    for k_outer, v_outer in all_sim_results.items():
        for k_inner, v_inner in v_outer.items():
            # If the inner value is a dictionary, flatten its contents with the combined key
            if isinstance(v_inner, dict):
                for k_deep, v_deep in v_inner.items():
                    flat_results_for_final_validation[f"{k_outer}.{k_inner}.{k_deep}"] = v_deep
            else:
                flat_results_for_final_validation[f"{k_outer}.{k_inner}"] = v_inner

    print("
" + "=" * 70)
    print("Final Comprehensive Validation Against Paper Claims")
    print("=" * 70)
    final_validation_passed = validate_simulation(flat_results_for_final_validation, final_expected_values, tolerance=0.10) # 10% tolerance for overall

    # Save all results to a JSON file
    results_filename = "data/all_simulation_results.json"
    os.makedirs('data', exist_ok=True)
    save_results(all_sim_results, results_filename)

    print(f"
Run finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    if final_validation_passed:
        print("All MHED-TOE simulations completed and passed final validation!")
        return 0 # Success
    else:
        print("MHED-TOE simulations completed, but some final validations failed.")
        return 1 # Failure

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
