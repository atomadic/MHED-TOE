"""
Integrated simulation across all MHED-TOE modules.
Demonstrates cross-validation between different pillars of the theory.
"""

import numpy as np
from .yukawa_axon_spectrum import YukawaAxonModel
from .orch_mt_coherence import OrchORCalculator
from .iit_holographic_phi import HolographicPhi
from .causal_set_sprinkling import CausalSetSprinkler
from .revelation_tensor import RevelationTensor
from .utils import validate_simulation

class IntegratedMHED:
    def __init__(self):
        self.yukawa = YukawaAxonModel(N=20)
        self.orch = OrchORCalculator(N_tubulins=100)
        self.phi = HolographicPhi(boundary_size=25)
        self.causal = CausalSetSprinkler(rho=10, volume=10)
        self.revelation = RevelationTensor()

    def run_all_simulations(self):
        print("
" + "=" * 50)
        print("Running Integrated MHED-TOE Simulation")
        print("=" * 50)

        yukawa_results = self.yukawa.main()
        orch_results = self.orch.main()
        phi_results = self.phi.main()
        causal_results = self.causal.main()
        revelation_results = self.revelation.main()

        all_results = {
            "yukawa_model": yukawa_results,
            "orch_calculator": orch_results,
            "holographic_phi": phi_results,
            "causal_sprinkler": causal_results,
            "revelation_tensor": revelation_results
        }

        # Perform integrated validation against key expected values
        expected_values = {
            "yukawa_model.fermion_masses.higgs_sim": 124.8, # GeV (accessing specific nested key)
            "orch_calculator.gamma_freq": 36.0, # Hz
            "holographic_phi.phi_holo": 2.847,
            "causal_sprinkler.u_hex_value": 2.84
        }

        # Flatten results for final comprehensive validation display
        flat_results_for_final_validation = {}
        for k_outer, v_outer in all_results.items():
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
        final_validation_passed = validate_simulation(flat_results_for_final_validation, expected_values, tolerance=0.10) # 10% tolerance for overall

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

if __name__ == "__main__":
    integrated_mhed = IntegratedMHED()
    all_sim_results, validation_status = integrated_mhed.run_all_simulations()
    print(f"
Integrated MHED-TOE simulation completed. All validations passed: {validation_status}")

