import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np
from collections import defaultdict
import random

# Import functions and classes (assuming they are in separate files)
from graph_d import ZDSubgraph
from norm_origin import normalize_origin

def find_leafs(graph):
    """
    Finds the leafs of a graph (vertices with degree < 2)
    Includes vertices with degree 0 as leafs
    """
    d = graph.d
    leafs = set()
    # Define directions for Z^2 (4-connected grid)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    # for all vertex check if it is a leaf
    for vert in graph.vertices:
        # initialize the degree as zero
        vert_deg = 0
        for e in directions:
            new_vert = tuple([vert[i] + e[i] for i in range(d)]) 
            # Check if edge exists in either direction (since edges are stored as sorted tuples)
            edge1 = tuple(sorted([vert, new_vert]))
            edge2 = tuple(sorted([new_vert, vert]))
            if edge1 in graph.edges or edge2 in graph.edges:
                vert_deg = vert_deg + 1

        if vert_deg < 2:
            # Count zero degree vertex as leafs too
            leafs.add(vert)

    return leafs

def plot_subgraphs_table(subgraphs, max_plots=20, figsize=(16, 12), title="ZDSubgraphs"):
    """
    Plot multiple ZDSubgraphs in a table layout
    
    Args:
        subgraphs: list of ZDSubgraph objects
        max_plots: maximum number of subgraphs to plot
        figsize: figure size
        title: overall figure title
    """
    n_plots = min(len(subgraphs), max_plots)
    
    # Calculate grid dimensions
    perf_n_rows = int(np.sqrt(3*n_plots)//2)
    n_rows = min(perf_n_rows, n_plots)  # Maximum 5 columns
    n_cols = (n_plots + n_rows - 1) // n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Remove spines from all subplots
    for ax in axes.flat:
        ax.set_frame_on(False)  # This removes the entire box
    
    # Handle case when there's only one subplot
    if n_plots == 1:
        axes = np.array([axes])
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes array for easy iteration
    axes_flat = axes.flatten()
    
    for idx in range(n_plots):
        ax = axes_flat[idx]
        (G,boundary) = subgraphs[idx]
        
        plot_single_subgraph(ax, G, boundary, n_plots, idx)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle(f"{title} (showing first {n_plots} of {len(subgraphs)})", fontsize=16)
    plt.tight_layout()
    # Create and show legend
    create_legend_demo()
    plt.show()



def plot_single_subgraph(ax, G, boundary, n_plots, idx=None):
    """
    Plot a single ZDSubgraph on given axes
    """
    if not G.vertices:
        ax.text(0.5, 0.5, "Empty Graph", ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return
    
    # Get bounds for the graph
    x_coords = [v[0] for v in G.vertices]
    y_coords = [v[1] for v in G.vertices]
    
    if not x_coords:  # Empty graph
        ax.text(0.5, 0.5, "Empty", ha='center', va='center')
        return
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Add some padding
    padding = 0.8
    x_range = max(max_x - min_x, 1)
    y_range = max(max_y - min_y, 1)
    
    # boundary vertices
    inner_vertex = G.vertices - boundary
    
    # Plot inner vertices (regular vertices)
    standard_size = min(20, 50/np.sqrt(n_plots))
    vertices_size = max(0.1, standard_size)
    if inner_vertex:
        inner_vert_x = [v[0] for v in inner_vertex]
        inner_vert_y = [v[1] for v in inner_vertex]
        ax.scatter(inner_vert_x, inner_vert_y, color='black', s=vertices_size, zorder=3, label='Inner vertex')
    
    # Plot boundary vertices with different color
    if boundary:
        boundary_x = [v[0] for v in boundary]
        boundary_y = [v[1] for v in boundary]
        ax.scatter(boundary_x, boundary_y, color='red', s=vertices_size, zorder=3, label='Boundary')
    
    # Plot internal edges (black)
    edge_size = min(max(0.2,standard_size//2),2)
    if G.edges:
        edges_list = []
        for edge in G.edges:
            edges_list.append([edge[0], edge[1]])
        
        lc = LineCollection(edges_list, colors='black', linewidths=edge_size, zorder=2)
        ax.add_collection(lc)
    
    # Plot boundary edges with different color (gray)
    boundary = G.get_boundary()
    if boundary:
        boundary_list = []
        for edge in boundary:
            boundary_list.append([edge[0], edge[1]])
        
        lc_boundary = LineCollection(boundary_list, colors='gray', linewidths=edge_size, 
                                   linestyle='solid', alpha=0.8, zorder=2)
        ax.add_collection(lc_boundary)
    
    # Set plot properties - remove grid and coordinates
    ax.set_aspect('equal')
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    
    # Remove grid and axis labels/ticks
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Add title with graph info
    title_parts = []
    if idx is not None:
        title_parts.append(f"Graph {idx+1}")
    title_parts.append(f"V:{len(G.vertices)}")
    title_parts.append(f"E:{len(G.edges)}")
    title_parts.append(f"B:{G.boundary_size()}")

    fontsize = min(9, standard_size)
    
    ax.set_title(' | '.join(title_parts), fontsize=fontsize)

def generate_random_subgraphs(n_graphs=20, max_size=6):
    """
    Generate random subgraphs for testing
    """
    graphs_leafs_pair = []
    
    for i in range(n_graphs):
        G = ZDSubgraph(d=2)
        
        # Random number of vertices (3 to 15)
        n_vertices = random.randint(3, 15)
        
        # Generate random connected component
        start_x, start_y = random.randint(0, max_size), random.randint(0, max_size)
        G.add_vertex((start_x, start_y))
        
        current_vertices = [(start_x, start_y)]
        
        for _ in range(n_vertices - 1):
            # Pick a random existing vertex to extend from
            base_vertex = random.choice(current_vertices)
            
            # Random direction
            direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            new_vertex = (base_vertex[0] + direction[0], base_vertex[1] + direction[1])
            
            # Ensure we stay within bounds
            if (0 <= new_vertex[0] <= max_size and 0 <= new_vertex[1] <= max_size):
                G.add_vertex(new_vertex)
                G.add_edge(base_vertex, new_vertex)
                current_vertices.append(new_vertex)
        
        # Normalize the graph
        G = normalize_origin(G,G.d)
        leafs = find_leafs(G)
        graphs_leafs_pair.append((G,leafs))
    
    return graphs_leafs_pair

def plot_connected_components_table(G, max_components=20, figsize=(15, 12)):
    """
    Plot each connected component of a graph as separate subgraphs in a table
    """
    # Get all connected components
    components = get_connected_components(G)
    
    # Take only the first max_components
    components = components[:max_components]
    
    # Create subgraphs for each component
    component_graphs = []
    for comp_vertices in components:
        comp_G = ZDSubgraph(d=2)
        # Add vertices
        for vertex in comp_vertices:
            comp_G.add_vertex(vertex)
        # Add edges that are within this component
        for edge in G.edges:
            if edge[0] in comp_vertices and edge[1] in comp_vertices:
                comp_G.add_edge(edge[0], edge[1])
        # Normalize
        comp_G = normalize_origin(comp_G,G.d)
        component_graphs.append(comp_G)
    
    return plot_subgraphs_table(component_graphs, max_plots=len(component_graphs), 
                               figsize=figsize, title="Connected Components")

def get_connected_components(G):
    """
    Extract all connected components from a graph
    """
    components_dict = defaultdict(list)
    
    # Group vertices by their root
    for vertex in G.vertices:
        root = G.uf.find(vertex)
        components_dict[root].append(vertex)
    
    return list(components_dict.values())

def create_legend_demo():
    """
    Create a separate plot showing the legend for all elements
    """
    
    # Create legend elements
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Inner Vertex'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Boundary Vertex'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='Internal Edge'),
        plt.Line2D([0], [0], color='gray', linewidth=2, label='Boundary Edge')
    ]
    
    plt.legend(handles=legend_elements, loc='lower right', ncol=3)


# Demonstration
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Generating random subgraphs with modified visualization...")
    
    # Generate random subgraphs
    random_graphs = generate_random_subgraphs(n_graphs=20, max_size=8)
    
    # Plot the subgraphs table
    fig1, axes1 = plot_subgraphs_table(random_graphs, title="ZDSubgraphs with Leaf Coloring")
    # Create and show legend
    legend_fig = create_legend_demo()
    plt.show()
    
    
   