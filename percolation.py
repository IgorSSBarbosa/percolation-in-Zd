import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import random
from graph_d import ZDSubgraph

class BondPercolation:
    def __init__(self, size=10, p=0.5):
        """
        Bond percolation on Z^2 grid
        
        Args:
            size: grid size (creates size x size grid)
            p: probability of keeping each bond
        """
        self.size = size
        self.p = p
        self.grid = ZDSubgraph(d=2)
        self.kept_edges = []
        self.removed_edges = []
        
    def generate_percolation(self):
        """Generate bond percolation configuration"""
        # Add all vertices in the grid
        for i in range(self.size + 1):  # +1 to include boundaries
            for j in range(self.size + 1):
                self.grid.add_vertex((i, j))
        
        # Generate horizontal bonds
        for i in range(self.size):
            for j in range(self.size + 1):
                if random.random() < self.p:
                    self.grid.add_edge((i, j), (i + 1, j))
                    self.kept_edges.append(((i, j), (i + 1, j)))
                else:
                    self.removed_edges.append(((i, j), (i + 1, j)))
        
        # Generate vertical bonds
        for i in range(self.size + 1):
            for j in range(self.size):
                if random.random() < self.p:
                    self.grid.add_edge((i, j), (i, j + 1))
                    self.kept_edges.append(((i, j), (i, j + 1)))
                else:
                    self.removed_edges.append(((i, j), (i, j + 1)))
    
    def get_largest_component(self):
        """Get the largest connected component"""
        component_sizes = self.grid.uf.get_component_sizes()
        if not component_sizes:
            return set()
        
        largest_root = max(component_sizes, key=component_sizes.get)
        largest_component = set()
        
        for coord_id, root in self.grid.uf.parent.items():
            if self.grid.uf.find(self.grid.uf._get_coord(coord_id)) == largest_root:
                coord = self.grid.uf._get_coord(coord_id)
                largest_component.add(coord)
        
        return largest_component
    
    def plot_percolation(self, show_largest_component=True, figsize=(10, 10)):
        """Plot the bond percolation configuration"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot vertices
        vertices = list(self.grid.vertices)
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        ax.scatter(x_coords, y_coords, color='black', s=30, zorder=3)
        
        # Plot kept edges (solid lines)
        if self.kept_edges:
            kept_lines = []
            for edge in self.kept_edges:
                kept_lines.append([edge[0], edge[1]])
            
            lc_kept = LineCollection(kept_lines, colors='blue', linewidths=2, zorder=2)
            ax.add_collection(lc_kept)
        
        # Plot removed edges (dashed red lines)
        if self.removed_edges:
            removed_lines = []
            for edge in self.removed_edges:
                removed_lines.append([edge[0], edge[1]])
            
            lc_removed = LineCollection(removed_lines, colors='red', linewidths=1, 
                                      linestyle='dashed', alpha=0.5, zorder=1)
            ax.add_collection(lc_removed)
        
        # Highlight largest component if requested
        if show_largest_component:
            largest_comp = self.get_largest_component()
            if largest_comp:
                lc_vertices = list(largest_comp)
                lc_x = [v[0] for v in lc_vertices]
                lc_y = [v[1] for v in lc_vertices]
                ax.scatter(lc_x, lc_y, color='green', s=50, zorder=4, 
                          label='Largest Component')
                
                # Highlight edges in largest component
                lc_edges = []
                for edge in self.kept_edges:
                    if edge[0] in largest_comp and edge[1] in largest_comp:
                        lc_edges.append([edge[0], edge[1]])
                
                if lc_edges:
                    lc_largest = LineCollection(lc_edges, colors='green', 
                                              linewidths=3, zorder=2.5)
                    ax.add_collection(lc_largest)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, self.size + 0.5)
        ax.set_ylim(-0.5, self.size + 0.5)
        ax.set_title(f'Bond Percolation on Z² (p={self.p}, size={self.size})')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.grid(True, alpha=0.3)
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Kept Bond'),
            plt.Line2D([0], [0], color='red', linewidth=1, linestyle='dashed', 
                      label='Removed Bond', alpha=0.7),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=8, label='Largest Component')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add statistics text
        stats_text = f'Kept bonds: {len(self.kept_edges)}\nRemoved bonds: {len(self.removed_edges)}\nComponents: {self.grid.connected_components()}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig, ax


def multiple_percolations_demo():
    """Demonstrate multiple percolation configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, p in enumerate([0.3, 0.5, 0.7, 0.9]):
        # Set random seed for reproducibility in demo
        random.seed(42 + i)
        
        # Generate and plot percolation
        percolation = BondPercolation(size=8, p=p)
        percolation.generate_percolation()
        
        # Plot on subplot
        ax = axes[i]
        vertices = list(percolation.grid.vertices)
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        ax.scatter(x_coords, y_coords, color='black', s=20, zorder=3)
        
        # Plot kept edges
        if percolation.kept_edges:
            kept_lines = []
            for edge in percolation.kept_edges:
                kept_lines.append([edge[0], edge[1]])
            lc_kept = LineCollection(kept_lines, colors='blue', linewidths=2, zorder=2)
            ax.add_collection(lc_kept)
        
        # Plot removed edges
        if percolation.removed_edges:
            removed_lines = []
            for edge in percolation.removed_edges:
                removed_lines.append([edge[0], edge[1]])
            lc_removed = LineCollection(removed_lines, colors='red', linewidths=1, 
                                      linestyle='dashed', alpha=0.5, zorder=1)
            ax.add_collection(lc_removed)
        
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, percolation.size + 0.5)
        ax.set_ylim(-0.5, percolation.size + 0.5)
        ax.set_title(f'p = {p}\nComponents: {percolation.grid.connected_components()}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create and visualize bond percolation with p=0.5
    print("Generating bond percolation with p=0.5...")
    percolation = BondPercolation(size=15, p=0.5)
    percolation.generate_percolation()
    
    # Print statistics
    print(f"Grid size: {percolation.size} x {percolation.size}")
    print(f"Number of vertices: {len(percolation.grid.vertices)}")
    print(f"Number of possible bonds: {len(percolation.kept_edges) + len(percolation.removed_edges)}")
    print(f"Kept bonds: {len(percolation.kept_edges)}")
    print(f"Removed bonds: {len(percolation.removed_edges)}")
    print(f"Connected components: {percolation.grid.connected_components()}")
    
    # Plot the result
    fig, ax = percolation.plot_percolation(show_largest_component=True)
    plt.show()
    
    # Demonstrate different probabilities
    print("\nGenerating comparison for different probabilities...")
    multiple_percolations_demo()