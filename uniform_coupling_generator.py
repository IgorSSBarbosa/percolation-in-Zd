import numpy as np
import itertools
from graph_d import ZDSubgraph
from plot_coupling_perc import plot_percolation

# Updated sample_bond_percolation function to ensure correct edge ordering
def sample_bond_percolation(d: int, L: int, random_state=None) -> list[float]:
    """
    Sample bond percolation random variables using uniform coupling.
    Returns edges in consistent order for plotting.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Calculate number of edges systematically
    if d == 1:
        num_edges = L

    else:
        # General formula for d dimensions
        num_edges = d * (L + 1)**(d - 1) * L
    
    # Generate uniform [0,1] random variables for each edge
    edge_variables = np.random.uniform(0, 1, num_edges)
    
    return edge_variables.tolist()

def ciclic_perm(v:list, n:int) -> list:
    """
    Permutes an array n times ciclically
    """
    s = len(v)
    n = n%s
    out = [0]*s
    for i in range(s):
        out[i] = v[i-n]

    return out

def convert_to_base_b(number, base):
    if not isinstance(number, int) or number < 0:
        raise ValueError("Number must be a non-negative integer.")
    if not isinstance(base, int) or base < 2:
        raise ValueError("Base must be an integer greater than or equal to 2.")

    if number == 0:
        return [0]

    digits = []
    while number > 0:
        remainder = number % base
        
        if remainder < 10:
            digits.append(remainder)
        else:
            digits.append(remainder)
        number //= base
    
    return digits


def get_edge_from_index(d: int, L: int, edge_index: int) -> tuple[tuple, tuple]:
    """
    Convert an edge index to a pair of nodes (start, end) in d dimensions.
    
    Args:
        d: Dimension of the lattice
        L: Linear size of the lattice (L+1 nodes in each dimension)
        edge_index: Index of the edge in the full list
        
    Returns:
        Tuple of two tuples representing the start and end nodes
    """
    # direction 
    axis_direction = edge_index % d
    vec_direction = [0]*d
    vec_direction[axis_direction]=1
    # atualize index
    index = edge_index// d

    # coordinates of the origin vertex of the edge
    origin_vertex = convert_to_base_b(index,L+1)

    gap = d - len(origin_vertex)
    if gap>0:
        for _ in range(gap):
            origin_vertex.append(0)

    start_node = tuple(ciclic_perm(origin_vertex, axis_direction+1))
    end_node = tuple(
            [start_node[i]+vec_direction[i] for i in range(d)]
        )

    return tuple([start_node, end_node])


def graph_structure(d: int, L: int, edges: list[float], p:float) -> ZDSubgraph:
    """
    Creates a graph structure of given sample of uniform coupling and a p parameter
    """
    assert len(edges)== d*L*(L+1)**(d-1)

    graph = ZDSubgraph(d=d)
    values = range(0,L)
    # Use itertools.product to generate all combinations
    vertices = list(itertools.product(values, repeat=d))
    for v in vertices:
        # add all vertex
        graph.add_vertex(v)

    
    for i,edge_value in enumerate(edges):
        if edge_value<1:
            edge = get_edge_from_index(d=d,L=L, edge_index=i)
            graph.add_edge(edge[0],edge[1])
    return graph


if __name__=="__main__":
    import matplotlib.pyplot as plt
    d=2
    L=4
    p = 0.5
    seed = 1
    edges_unif = sample_bond_percolation(d=d, L=L,random_state=seed)

    G = graph_structure(d=d, L=L, edges=edges_unif, p=p)
    if d<4:
        plot_percolation(edges_unif, d=d, L=L, p=p)
        plt.show()