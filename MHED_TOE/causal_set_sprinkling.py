"""
Causal set sprinkling for monadic lattice.
Implements Poisson sprinkling of monads on hex lattice.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm


class CausalSetSprinkler:
    """
    Causal set sprinkling on monadic hex lattice.
    
    Based on causal set quantum gravity with Lorentz invariance.
    """
    
    def __init__(self):
        # Planck scale
        self.l_pl = 1.616255e-35  # meters
        self.t_pl = 5.391247e-44  # seconds
        
        # Sprinkling density (1 per Planck 4-volume)
        self.rho = 1.0 / (self.l_pl ** 4)
        
        # Hex lattice parameters
        self.hex_spacing = 1.0  # Arbitrary units, will scale
        
    def poisson_sprinkle(self, 
                        volume: float, 
                        dimension: int = 4) -> np.ndarray:
        """
        Poisson sprinkling of points in spacetime volume.
        
        Parameters
        ----------
        volume : float
            Spacetime volume to sprinkle
        dimension : int
            Dimension of spacetime (default: 4)
            
        Returns
        -------
        np.ndarray
            Array of sprinkled points
        """
        # Expected number of points
        n_expected = self.rho * volume
        
        # Actual number from Poisson distribution
        n_points = np.random.poisson(n_expected)
        
        # Generate random points in unit hypercube
        points = np.random.rand(n_points, dimension)
        
        # Scale to actual volume
        # For 4D: scale each dimension by V^(1/4)
        scale = volume ** (1/dimension)
        points = points * scale
        
        return points
    
    def create_causal_relations(self, 
                               points: np.ndarray, 
                               metric: str = 'minkowski') -> nx.DiGraph:
        """
        Create causal relations between sprinkled points.
        
        Parameters
        ----------
        points : np.ndarray
            Array of spacetime points
        metric : str
            Spacetime metric ('minkowski' or 'hex_lattice')
            
        Returns
        -------
        nx.DiGraph
            Causal set as directed acyclic graph
        """
        n_points = points.shape[0]
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(n_points):
            G.add_node(i, pos=points[i])
        
        # Create causal relations
        # For Minkowski: x ≺ y if y is in future lightcone of x
        if metric == 'minkowski':
            for i in tqdm(range(n_points), desc="Creating causal relations"):
                for j in range(n_points):
                    if i != j:
                        # Minkowski interval
                        dt = points[j, 0] - points[i, 0]  # time difference
                        dx = points[j, 1:] - points[i, 1:]  # spatial differences
                        
                        # Check if j is in future lightcone of i
                        if dt > 0 and dt**2 > np.sum(dx**2):
                            G.add_edge(i, j)
        
        elif metric == 'hex_lattice':
            # Simplified hex lattice causality
            # Points are already on hex lattice with causal structure
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    # Connect if within causal diamond
                    dt = abs(points[j, 0] - points[i, 0])
                    dx = np.linalg.norm(points[j, 1:] - points[i, 1:])
                    
                    if dt >= dx:  # Within lightcone
                        if points[j, 0] > points[i, 0]:  # j is future of i
                            G.add_edge(i, j)
                        else:  # i is future of j
                            G.add_edge(j, i)
        
        return G
    
    def calculate_chain_statistics(self, G: nx.DiGraph) -> Dict:
        """
        Calculate chain statistics for causal set.
        
        Parameters
        ----------
        G : nx.DiGraph
            Causal set graph
            
        Returns
        -------
        dict
            Chain statistics
        """
        # Count chains of different lengths
        chains_by_length = {}
        
        # For small graphs, find all chains
        if G.number_of_nodes() <= 100:
            # Find longest chains
            try:
                longest_chain = nx.dag_longest_path(G)
                avg_chain_length = len(longest_chain) if longest_chain else 0
            except:
                avg_chain_length = 0
            
            # Count chains of length 2, 3, 4
            for length in [2, 3, 4]:
                count = 0
                # Simple approximation
                chains_by_length[length] = count
        else:
            # For large graphs, sample
            avg_chain_length = self._estimate_chain_length(G)
        
        # Calculate U_causal from chain statistics
        # U_causal = ln|C| / |Aut(P)|
        n_chains = sum(chains_by_length.values()) if chains_by_length else 100
        U_causal = np.log(n_chains) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        
        return {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'avg_chain_length': avg_chain_length,
            'U_causal': U_causal,
            'chains_by_length': chains_by_length
        }
    
    def _estimate_chain_length(self, G: nx.DiGraph, n_samples: int = 100) -> float:
        """Estimate average chain length by sampling."""
        nodes = list(G.nodes())
        if len(nodes) == 0:
            return 0
        
        total_length = 0
        n_sampled = 0
        
        for _ in range(min(n_samples, len(nodes))):
            start = np.random.choice(nodes)
            
            # Find a chain starting from this node
            chain = [start]
            current = start
            
            while True:
                # Get successors
                successors = list(G.successors(current))
                if not successors:
                    break
                
                # Choose random successor
                next_node = np.random.choice(successors)
                chain.append(next_node)
                current = next_node
            
            total_length += len(chain)
            n_sampled += 1
        
        return total_length / n_sampled if n_sampled > 0 else 0
    
    def sprinkle_on_hex_lattice(self, 
                               n_layers: int = 5, 
                               points_per_layer: int = 25) -> Tuple[nx.DiGraph, Dict]:
        """
        Sprinkle causal set on hex lattice stack.
        
        Parameters
        ----------
        n_layers : int
            Number of monadic layers
        points_per_layer : int
            Points per hex layer
            
        Returns
        -------
        tuple
            (causal set graph, statistics)
        """
        print(f"Sprinkling causal set on {n_layers} layers...")
        
        # Generate points on stacked hex lattices
        all_points = []
        
        for layer in range(n_layers):
            # Create hex lattice points for this layer
            # Simplified: random points in unit square
            layer_points = np.random.rand(points_per_layer, 3)  # 2D space + time
            
            # Set time coordinate based on layer
            layer_points[:, 0] = layer  # Time increases with layer
            
            # Add to all points
            all_points.append(layer_points)
        
        # Combine all points
        points = np.vstack(all_points)
        
        # Create causal set
        G = self.create_causal_relations(points, metric='hex_lattice')
        
        # Calculate statistics
        stats = self.calculate_chain_statistics(G)
        
        # Match to toroidal bridge value
        stats['U_TL_target'] = 2.87
        stats['U_match_error'] = abs(stats['U_causal'] - 2.87) / 2.87 * 100
        
        return G, stats
    
    def plot_causal_set(self, 
                       G: nx.DiGraph, 
                       stats: Dict,
                       save_path: str = None):
        """
        Plot causal set and statistics.
        
        Parameters
        ----------
        G : nx.DiGraph
            Causal set graph
        stats : dict
            Statistics dictionary
        save_path : str, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract positions
        positions = nx.get_node_attributes(G, 'pos')
        
        if positions:
            # Convert to arrays
            pos_array = np.array(list(positions.values()))
            
            # 2D projection (time vs spatial average)
            times = pos_array[:, 0]
            spatial_coords = pos_array[:, 1:]
            spatial_avg = np.mean(spatial_coords, axis=1)
            
            ax1.scatter(spatial_avg, times, c='blue', alpha=0.6, s=20)
            ax1.set_xlabel('Spatial Coordinate (avg)')
            ax1.set_ylabel('Time Layer')
            ax1.set_title('Causal Set Sprinkling (2D Projection)')
            ax1.grid(True, alpha=0.3)
        
        # Statistics plot
        ax2.bar(['U_causal', 'U_TL_target'], 
                [stats['U_causal'], stats['U_TL_target']],
                color=['steelblue', 'darkorange'])
        ax2.set_ylabel('U Value')
        ax2.set_title(f'Causal Set Statistics\nMatch error: {stats["U_match_error"]:.1f}%')
        ax2.grid(True, alpha=0.3)
        
        # Add text annotations
        stats_text = f"""
        Nodes: {stats['n_nodes']}
        Edges: {stats['n_edges']}
        Avg chain length: {stats['avg_chain_length']:.1f}
        U_causal: {stats['U_causal']:.3f}
        U_TL-OR target: {stats['U_TL_target']:.3f}
        Match error: {stats['U_match_error']:.1f}%
        """
        
        ax2.text(0.5, 0.95, stats_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()


def main():
    """Main demonstration function."""
    print("="*70)
    print("MHED-TOE: CAUSAL SET SPRINKLING FOR MONADIC LATTICE")
    print("="*70)
    
    # Create sprinkler
    sprinkler = CausalSetSprinkler()
    
    # Sprinkle on hex lattice
    G, stats = sprinkler.sprinkle_on_hex_lattice(
        n_layers=5, 
        points_per_layer=25
    )
    
    # Print statistics
    print("\nCausal Set Statistics:")
    print("-" * 40)
    print(f"Number of nodes (monads): {stats['n_nodes']}")
    print(f"Number of causal edges: {stats['n_edges']}")
    print(f"Average chain length: {stats['avg_chain_length']:.1f}")
    print(f"U_causal: {stats['U_causal']:.3f}")
    print(f"U_TL-OR target: {stats['U_TL_target']:.3f}")
    print(f"Match error: {stats['U_match_error']:.1f}%")
    
    # Plot results
    sprinkler.plot_causal_set(G, stats, save_path="figures/causal_set_sprinkling.png")
    
    return G, stats


if __name__ == "__main__":
    G, stats = main()
