import numpy as np
from typing import List

# Updated sample_bond_percolation function to ensure correct edge ordering
def sample_bond_percolation(d: int, L: int, random_state=None) -> List[float]:
    """
    Sample bond percolation random variables using uniform coupling.
    Returns edges in consistent order for plotting.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Calculate number of edges systematically
    if d == 1:
        num_edges = L
    elif d == 2:
        num_edges = (L + 1) * L + L * (L + 1)  # horizontal + vertical
    elif d == 3:
        num_edges = (L + 1)**2 * L * 3  # edges in x, y, z directions
    else:
        # General formula for d dimensions
        num_edges = d * (L + 1)**(d - 1) * L
    
    # Generate uniform [0,1] random variables for each edge
    edge_variables = np.random.uniform(0, 1, num_edges)
    
    return edge_variables.tolist()
