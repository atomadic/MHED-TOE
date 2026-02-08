"""
Holographic integrated information (Φ) calculations.
Computes Φ_holo from Ryu-Takayanagi surfaces on hex CFT boundaries.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import itertools
from scipy import sparse


class HolographicPhi:
    """
    Calculate holographic integrated information Φ from RT surfaces.
    
    Based on AdS3/CFT2 duality on hex lattice boundaries.
    """
    
    def __init__(self):
        # Constants
        self.G_N = 6.67430e-11  # m³/kg/s² (Newton's constant)
        self.l_pl = 1.616255e-35  # m (Planck length)
        
        # Default parameters
        self.default_n_nodes = 25  # 5x5 hex lattice
        self.default_temperature = 0.01  # Thermal state parameter
    
    def create_hex_lattice(self, n: int = 5, m: int = 5) -> nx.Graph:
        """
        Create hexagonal lattice graph.
        
        Parameters
        ----------
        n, m : int
            Lattice dimensions
            
        Returns
        -------
        nx.Graph
            Hexagonal lattice
        """
        G = nx.hexagonal_lattice_graph(n, m)
        
        # Label nodes for easier access
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['id'] = i
            
        return G
    
    def calculate_rt_surface(self, 
                           G: nx.Graph, 
                           subset: List[int]) -> Tuple[float, List]:
        """
        Calculate Ryu-Takayanagi minimal surface for subset A.
        
        Parameters
        ----------
        G : nx.Graph
            Boundary CFT graph
        subset : list of int
            Nodes in region A
            
        Returns
        -------
        tuple
            (area, min_cut_edges)
        """
        # Complement of A
        all_nodes = list(G.nodes())
        complement = [node for node in all_nodes if node not in subset]
        
        if not complement:  # Whole boundary
            return 0.0, []
        
        # Minimum cut between A and complement
        try:
            cut_value, partition = nx.minimum_cut(G, subset, complement)
            # Area is proportional to cut value
            area = cut_value / 4.0  # In Planck units (Area/4G_N)
        except:
            # Fallback: use edge boundary
            boundary_edges = []
            for node in subset:
                for neighbor in G.neighbors(node):
                    if neighbor not in subset:
                        boundary_edges.append((node, neighbor))
            area = len(boundary_edges) / 4.0
        
        return area, []
    
    def calculate_phi_holo(self, 
                          G: nx.Graph = None,
                          max_subset_size: int = 12) -> Dict:
        """
        Calculate holographic Φ for all subsets.
        
        Parameters
        ----------
        G : nx.Graph, optional
            CFT boundary graph
        max_subset_size : int
            Maximum subset size to consider
            
        Returns
        -------
        dict
            Φ calculation results
        """
        if G is None:
            G = self.create_hex_lattice(5, 5)
        
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        
        results = {
            'sizes': [],
            'areas': [],
            'phi_values': [],
            'max_phi': 0.0,
            'optimal_subset': None,
            'optimal_size': 0
        }
        
        # Consider subsets of different sizes
        for size in range(1, min(max_subset_size, n_nodes // 2) + 1):
            # Sample random connected subsets (for efficiency)
            # In full calculation, would consider all subsets
            best_phi = 0.0
            best_area = 0.0
            best_subset = None
            
            # Sample some subsets of this size
            n_samples = min(100, comb(len(nodes), size, exact=True))
            for _ in range(n_samples):
                subset = list(np.random.choice(nodes, size, replace=False))
                
                # Ensure connectivity (approximate)
                if not self._is_connected_subgraph(G, subset):
                    continue
                
                # Calculate RT surface area
                area, _ = self.calculate_rt_surface(G, subset)
                
                # Φ_holo = S_RT / log|A|
                if size > 1:
                    phi = area / np.log(size)
                else:
                    phi = 0.0
                
                if phi > best_phi:
                    best_phi = phi
                    best_area = area
                    best_subset = subset
            
            if best_subset is not None:
                results['sizes'].append(size)
                results['areas'].append(best_area)
                results['phi_values'].append(best_phi)
                
                if best_phi > results['max_phi']:
                    results['max_phi'] = best_phi
                    results['optimal_subset'] = best_subset
                    results['optimal_size'] = size
        
        return results
    
    def _is_connected_subgraph(self, G: nx.Graph, subset: List) -> bool:
        """Check if subset forms connected subgraph."""
        if len(subset) <= 1:
            return True
        
        # Create subgraph
        subgraph = G.subgraph(subset)
        return nx.is_connected(subgraph)
    
    def scale_to_brain(self, phi_small: float, n_neurons: int = 8.6e10) -> float:
        """
        Scale Φ from small lattice to brain scale.
        
        Parameters
        ----------
        phi_small : float
            Φ for small lattice
        n_neurons : int
            Number of neurons in brain
            
        Returns
        -------
        float
            Scaled Φ for brain
        """
        # Scaling: Φ ∝ N log N for integrated systems
        n_small = self.default_n_nodes
        
        # Log scaling from IIT
        phi_brain = phi_small * (np.log(n_neurons) / np.log(n_small))
        
        return phi_brain
    
    def plot_phi_vs_size(self, results: Dict, save_path: str = None):
        """
        Plot Φ vs subset size.
        
        Parameters
        ----------
        results : dict
            Φ calculation results
        save_path : str, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Φ vs size
        ax1.plot(results['sizes'], results['phi_values'], 'bo-', linewidth=2)
        ax1.axhline(y=results['max_phi'], color='r', linestyle='--',
                   label=f'Max Φ = {results["max_phi"]:.3f}')
        ax1.axvline(x=results['optimal_size'], color='g', linestyle='--',
                   label=f'Optimal size = {results["optimal_size"]}')
        
        ax1.set_xlabel('Subset Size |A|')
        ax1.set_ylabel('Φ_holo')
        ax1.set_title('Holographic Φ vs Subset Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Area vs size
        ax2.plot(results['sizes'], results['areas'], 'ro-', linewidth=2)
        ax2.set_xlabel('Subset Size |A|')
        ax2.set_ylabel('RT Surface Area')
        ax2.set_title('Ryu-Takayanagi Area vs Subset Size')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        # Print results
        print("\nHolographic Φ Results:")
        print("-" * 40)
        print(f"Maximum Φ_holo: {results['max_phi']:.3f}")
        print(f"Optimal subset size: {results['optimal_size']}")
        print(f"Scaled to brain (N={8.6e10:.0e} neurons):")
        phi_brain = self.scale_to_brain(results['max_phi'])
        print(f"  Φ_brain ≈ {phi_brain:.0f}")
        print(f"  (Human brain Φ estimates: 10³-10⁴)")


def main():
    """Main demonstration function."""
    print("="*70)
    print("MHED-TOE: HOLOGRAPHIC INTEGRATED INFORMATION CALCULATOR")
    print("="*70)
    
    # Create calculator
    phi_calc = HolographicPhi()
    
    # Create hex lattice
    G = phi_calc.create_hex_lattice(5, 5)
    print(f"Created hexagonal lattice with {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Calculate Φ
    results = phi_calc.calculate_phi_holo(G, max_subset_size=12)
    
    # Plot results
    phi_calc.plot_phi_vs_size(results, save_path="figures/holographic_phi.png")
    
    # Calculate brain scaling
    phi_brain = phi_calc.scale_to_brain(results['max_phi'])
    print(f"\nScaled to human brain: Φ ≈ {phi_brain:.0f}")
    
    return results


if __name__ == "__main__":
    results = main()
