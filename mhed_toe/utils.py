"""
Utility functions and constants for MHED-TOE.
"""

import numpy as np
import json
from typing import Dict, Any, List, Union
from dataclasses import dataclass

@dataclass
class Constants:
    hbar: float = 1.054571817e-34
    G: float = 6.67430e-11
    c: float = 2.99792458e8
    k_B: float = 1.380649e-23
    e: float = 1.602176634e-19

def validate_simulation(results_dict: Dict[str, Any],
                        expected_values: Dict[str, float] = None,
                        tolerance: float = 0.1) -> bool:
    print("=" * 40)
    print("Simulation Validation")
    print("=" * 40)
    
    if expected_values is None:
        print("No expected values provided for validation.")
        return True

    all_passed = True
    for key, expected in expected_values.items():
        if key in results_dict:
            actual = results_dict[key]
            if isinstance(actual, (int, float, np.number)):
                # Calculate percentage error for numerical values
                if expected == 0:
                    error_percent = float('inf') if actual != 0 else 0.0
                else:
                    error_percent = abs((actual - expected) / expected) * 100
                
                if error_percent <= tolerance * 100:
                    print(f"✅ {key}: Actual={actual:.4g}, Expected={expected:.4g}, Error={error_percent:.2f}% (within {tolerance*100:.0f}% tolerance)")
                else:
                    print(f"❌ {key}: Actual={actual:.4g}, Expected={expected:.4g}, Error={error_percent:.2f}% (exceeds {tolerance*100:.0f}% tolerance)")
                    all_passed = False
            else:
                print(f"⚠️ {key}: Skipping validation for non-numerical value (type: {type(actual).__name__}). Actual={actual}")
        else:
            print(f"⚠️ {key}: Expected value not found in results.")

    print("=" * 40)
    if all_passed:
        print("All critical validations passed!")
    else:
        print("Some validations failed. Review results.")
    print("=" * 40)
    return all_passed

def save_results(results: Dict[str, Any],
                 filename: str = "data/results.json") -> None:
    import json
    import numpy as np

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, cls=NpEncoder)
    print(f"Results saved to {filename}")

def load_results(filename: str = "data/results.json") -> Dict[str, Any]:
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_error(measured: float, predicted: float) -> float:
    if measured == 0:
        return float('inf') if predicted != 0 else 0
    return abs(predicted - measured) / abs(measured) * 100

def monte_carlo_error_propagation(func, params: Dict[str, float],
                                  uncertainties: Dict[str, float],
                                  n_samples: int = 10000) -> tuple:
    samples = []
    for _ in range(n_samples):
        sample_params = {k: np.random.normal(v, uncertainties.get(k, 0)) for k, v in params.items()}
        samples.append(func(**sample_params))
    return np.mean(samples), np.std(samples)

def latex_table(data: List[List[Union[str, float]]],
                headers: List[str],
                caption: str = "",
                label: str = "") -> str:
    n_cols = len(headers)
    
    # Use explicit newlines '
' in Python string for newlines in the output file.
    # Use '\' in Python string to produce a single '' in the output file for LaTeX commands.
    # Use '{{' and '}}' to escape curly braces that are part of the f-string literal, but actual curly braces in the LaTeX output.
    table_str = (
        f"\begin{{table}}[ht]
" # Corrected
        f"\centering
" # Corrected
        f"\caption{{{caption}}}
" # Corrected
        f"\label{{{label}}}
" # Corrected
        f"\begin{{tabular}}{{@{{{('l' * n_cols)}}}}}}\n" # Corrected
        f"\toprule
" # Corrected
    )
    # Each header item separated by ' & ', followed by LaTeX newline (\), then Python newline (
)
    table_str += " & ".join(headers) + " \\n\midrule\n" # Corrected

    for row in data:
        formatted_row = [f'{{x:.2f}}' if isinstance(x, (int, float)) else str(x) for x in row]
        # Each row item separated by ' & ', followed by LaTeX newline (\), then Python newline (
)
        table_str += " & ".join(formatted_row) + " \\n" # Corrected

    table_str += (
        f"\bottomrule
" # Corrected
        f"\end{{tabular}}
" # Corrected
        f"\end{{table}}
" # Corrected
    )
    return table_str

def generate_summary_stats(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results_list:
        return {}

    summary = {}
    # Example: Summarize numerical results from multiple runs
    for key in results_list[0].keys():
        if isinstance(results_list[0][key], (int, float, np.number)):
            values = [res[key] for res in results_list if key in res and isinstance(res[key], (int, float, np.number))]
            if values:
                summary[key + '_mean'] = np.mean(values)
                summary[key + '_std'] = np.std(values)
        elif isinstance(results_list[0][key], list) and results_list[0][key] and isinstance(results_list[0][key][0], (int, float, np.number)):
            # Handle lists of numerical values, e.g., error arrays
            all_list_values = []
            for res in results_list:
                if key in res and isinstance(res[key], list):
                    all_list_values.extend([v for v in res[key] if isinstance(v, (int, float, np.number))])
            if all_list_values:
                summary[key + '_mean'] = np.mean(all_list_values)
                summary[key + '_std'] = np.std(all_list_values)

    return summary

if __name__ == "__main__":
    print("MHED-TOE Utilities")
    print("=" * 40)
    const = Constants()
    print(f"Planck constant (hbar): {const.hbar:.2e} J.s")

    # Example validation
    test_results = {
        "higgs_mass": 125.0,
        "eeg_freq": 36.1,
        "phi_holo": 2.85
    }
    expected = {
        "higgs_mass": 125.09,
        "eeg_freq": 36.0,
        "phi_holo": 2.847
    }
    validate_simulation(test_results, expected, tolerance=0.01)

    # Example Monte Carlo Error Propagation
    def sample_func(a, b):
        return a * b

    params = {"a": 10.0, "b": 5.0}
    uncertainties = {"a": 0.1, "b": 0.05}
    mean_val, std_dev = monte_carlo_error_propagation(sample_func, params, uncertainties)
    print(f"
Monte Carlo Error Propagation for a*b: Mean={mean_val:.2f}, Std Dev={std_dev:.2f}")
