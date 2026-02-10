"""
Basic usage example for MHED-TOE.
Demonstrates all major components of the framework.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from mhed_toe import YukawaAxonModel, OrchORCalculator, HolographicPhi
from mhed_toe import CausalSetSprinkler, RevelationTensor, IntegratedMHED
from mhed_toe.utils import Constants, validate_simulation

def main():
    print("=" * 60)
    print("MHED-TOE: Basic Usage Example")
    print("=" * 60)

    # Initialize individual components
    yukawa_model = YukawaAxonModel()
    orch_calc = OrchORCalculator()
    phi_calc = HolographicPhi()
    causal_sprinkler = CausalSetSprinkler()
    revelation_tensor = RevelationTensor()
    constants = Constants()

    print("
--- Yukawa Axon Model ---")
    yukawa_results = yukawa_model.main()

    print("
--- Orch-OR Calculator ---")
    orch_results = orch_calc.main()

    print("
--- Holographic Phi ---")
    phi_results = phi_calc.main()

    print("
--- Causal Set Sprinkler ---")
    causal_results = causal_sprinkler.main()

    print("
--- Revelation Tensor ---")
    revelation_results = revelation_tensor.main()

    print("
--- Integrated Simulation ---")
    integrated_mhed = IntegratedMHED()
    all_sim_results, validation_status = integrated_mhed.run_all_simulations()

    print("
Basic usage example completed successfully!")
    return all_sim_results

if __name__ == '__main__':
    results = main()
