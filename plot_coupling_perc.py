import matplotlib.pyplot as plt

def plot_percolation(edges: list[float], d: int, L: int, p: float):
    """
    Plot bond percolation configuration for dimensions 1, 2, and 3.
    
    Parameters:
    -----------
    edges : List[float]
        List of uniform [0,1] random variables from sample_bond_percolation()
    d : int
        Dimension of the lattice (1, 2, or 3)
    L : int
        Size of the box [0,L]^d
    p : float
        Probability threshold for bonds to be open
    """
    if d not in [1, 2, 3]:
        raise ValueError("Dimension d must be 1, 2, or 3 for visualization")
    
    fig = plt.figure(figsize=(10, 8))
    
    if d == 1:
        _plot_1d_percolation(edges, L, p, fig)
    elif d == 2:
        _plot_2d_percolation(edges, L, p, fig)
    elif d == 3:
        _plot_3d_percolation(edges, L, p, fig)


def _plot_1d_percolation(edges: list[float], L: int, p: float, fig):
    """Plot 1D percolation"""
    ax = fig.add_subplot(111)
    
    # In 1D, vertices are points on a line, edges connect consecutive points
    vertices = list(range(L + 1))
    open_edges = []
    closed_edges = []
    
    edge_index = 0
    for i in range(L):
        if edges[edge_index] <= p:
            open_edges.append((i, i + 1))
        else:
            closed_edges.append((i, i + 1))
        edge_index += 1
    
    # Plot vertices
    ax.scatter(vertices, [0] * len(vertices), color='black', s=20, zorder=1)
    
    # Plot open edges (black)
    for edge in open_edges:
        ax.plot([edge[0], edge[1]], [0, 0], color='black', linewidth=2, zorder=1)
    
    # Plot closed edges (gray)
    for edge in closed_edges:
        ax.plot([edge[0], edge[1]], [0, 0], color='gray', linewidth=1, zorder=1)
    
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')
    ax.set_title(f'1D Bond Percolation (L={L}, p={p})')
    ax.set_xlabel('Position')
    ax.set_yticks([])
    
    # Add legend
    ax.plot([], [], color='black', linewidth=3, label='Open bond')
    ax.plot([], [], color='gray', linewidth=2, label='Closed bond')
    ax.legend()

def _plot_2d_percolation(edges: list[float], L: int, p: float, fig):
    """Plot 2D percolation"""
    ax = fig.add_subplot(111)
    
    # Generate all vertices in 2D grid
    vertices = [(i, j) for i in range(L + 1) for j in range(L + 1)]
    
    open_edges = []
    closed_edges = []
    
    edge_index = 0
    # Horizontal edges
    for i in range(L + 1):
        for j in range(L):
            if edges[edge_index] <= p:
                open_edges.append(((i, j), (i, j + 1)))
            else:
                closed_edges.append(((i, j), (i, j + 1)))
            edge_index += 1
    
    # Vertical edges  
    for i in range(L):
        for j in range(L + 1):
            if edges[edge_index] <= p:
                open_edges.append(((i, j), (i + 1, j)))
            else:
                closed_edges.append(((i, j), (i + 1, j)))
            edge_index += 1
    
    # Plot vertices
    x_vals = [v[0] for v in vertices]
    y_vals = [v[1] for v in vertices]
    ax.scatter(x_vals, y_vals, color='black', s=30, zorder=3)
    
    # Plot open edges (on top)
    for edge in open_edges:
        ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 
                color='black', linewidth=1, zorder=1)
    
    # Plot closed edges first (behind)
    for edge in closed_edges:
        ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 
                color='gray', linewidth=1, alpha=0.4, zorder=1)
    
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.5, L + 0.5)
    ax.set_aspect('equal')
    ax.set_title(f'2D Bond Percolation (L={L}, p={p})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Add legend
    ax.plot([], [], color='black', linewidth=2, label='Open bond')
    ax.plot([], [], color='gray', linewidth=1, alpha=0.4, label='Closed bond')
    ax.legend()

def _plot_3d_percolation(edges: list[float], L: int, p: float, fig):
    """Plot 3D percolation"""
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate all vertices in 3D grid
    vertices = [(i, j, k) for i in range(L + 1) for j in range(L + 1) for k in range(L + 1)]
    
    open_edges = []
    closed_edges = []
    
    edge_index = 0
    # Generate edges in all three directions
    for i in range(L + 1):
        for j in range(L + 1):
            for k in range(L + 1):
                # x-direction edges
                if i < L:
                    if edges[edge_index] <= p:
                        open_edges.append(((i, j, k), (i + 1, j, k)))
                    else:
                        closed_edges.append(((i, j, k), (i + 1, j, k)))
                    edge_index += 1
                
                # y-direction edges
                if j < L:
                    if edges[edge_index] <= p:
                        open_edges.append(((i, j, k), (i, j + 1, k)))
                    else:
                        closed_edges.append(((i, j, k), (i, j + 1, k)))
                    edge_index += 1
                
                # z-direction edges
                if k < L:
                    if edges[edge_index] <= p:
                        open_edges.append(((i, j, k), (i, j, k + 1)))
                    else:
                        closed_edges.append(((i, j, k), (i, j, k + 1)))
                    edge_index += 1
    
    # Plot vertices
    x_vals = [v[0] for v in vertices]
    y_vals = [v[1] for v in vertices]
    z_vals = [v[2] for v in vertices]
    ax.scatter(x_vals, y_vals, z_vals, color='black', s=20, alpha=0.6)
    
    # Plot closed edges
    # for edge in closed_edges[:min(200, len(closed_edges))]:  # Limit for clarity
    #     ax.plot([edge[0][0], edge[1][0]], 
    #             [edge[0][1], edge[1][1]], 
    #             [edge[0][2], edge[1][2]], 
    #             'gray-', linewidth=1, alpha=0.3)
    
    # Plot open edges
    for edge in open_edges[:min(200, len(open_edges))]:  # Limit for clarity
        ax.plot([edge[0][0], edge[1][0]], 
                [edge[0][1], edge[1][1]], 
                [edge[0][2], edge[1][2]], 
                color='black', linewidth=2, alpha=0.8)
    
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.5, L + 0.5)
    ax.set_zlim(-0.5, L + 0.5)
    ax.set_title(f'3D Bond Percolation (L={L}, p={p})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add legend manually since 3D plots don't support legend well
    ax.text2D(0.05, 0.95, "Black: Open bonds", 
              transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3"))
    

# Example usage
if __name__ == "__main__":
    from uniform_coupling_generator import sample_bond_percolation

    # Test different dimensions
    for dim in [1, 2, 3]:
        print(f"\n=== Testing {dim}D Percolation ===")
        
        L_size = 4 if dim == 3 else 6  # Smaller for 3D for clarity
        
        edges = sample_bond_percolation(dim, L_size, random_state=42)
        print(f"Number of edges in {dim}D: {len(edges)}")
        
        # Plot with different probabilities
        for prob in [0.3, 0.6]:
            plot_percolation(edges, dim, L_size, prob)

    plt.tight_layout()
    plt.show()