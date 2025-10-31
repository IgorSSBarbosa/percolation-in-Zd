import argparse
import numpy as np
from graph_d import ZDSubgraph
from norm_origin import normalize_origin
from Z2_plot import plot_subgraphs_table
from tqdm import tqdm

class cluster_number_density:
    def __init__(self, args):
        self.s = args["max_cluster_size"]
        self.d = args["dimension"]
        self.plot = args["plot"]
        self.max_plots = args["max_plots"]
        
        # Construct a base graph with just the origin
        graph = ZDSubgraph(d=self.d)
        origin = tuple([0]*self.d)
        graph.add_vertex(origin)

        # Create a dictionary for all the possible clusters
        self.size_to_clusters_boundaries_pair = dict()

        # For each cluster we associate with its boundary, i.e. vertices which can recieve more neighbor vertex
        boundary = frozenset([origin])  # Use frozenset for hashability
        
        # Store as tuple (graph, boundary) - both need to be hashable
        initial_cluster_boundary_pair = set()
        initial_cluster_boundary_pair.add((self._graph_to_tuple(graph), boundary))
        self.size_to_clusters_boundaries_pair[1] = initial_cluster_boundary_pair  # Size 1, not 0

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--plot",
            action='store_true',
            help='When add this argument plots a visualization of the clusters, only available in dimension 2',
        )
        parser.add_argument(
            "--max_plots",
            default=20,
            type=int,
            help='The maximum number of clusters that will be visualized',
        )
        parser.add_argument(
            "--dimension",
            "-d",
            type=int,
            default=2,
            help='The dimention in Z^d',
        )
        parser.add_argument(
            "--max_cluster_size",
            "-s",
            default=5,
            type=int,
            help="The maximum size of cluster will be calculated",
        )

    def _graph_to_tuple(self, graph):
        """Convert graph to hashable tuple representation"""
        vertices = tuple(sorted(graph.vertices))
        edges = tuple(sorted(tuple(sorted(edge)) for edge in graph.edges))
        return (vertices, edges)

    def _tuple_to_graph(self, graph_tuple):
        """Convert tuple representation back to ZDSubgraph"""
        vertices, edges = graph_tuple
        graph = ZDSubgraph(d=self.d)
        for vertex in vertices:
            graph.add_vertex(vertex)
        for edge in edges:
            graph.add_edge(edge[0], edge[1])
        return graph

    def is_a_inner_vertex(self, vertex, cluster_tuple):
        """Check if a vertex is in the boundary in the cluster"""
        cluster = self._tuple_to_graph(cluster_tuple)
        neighbor_count = 0
        
        for e in cluster.directions:
            new_vertex = tuple(vertex[i] + e[i] for i in range(self.d))
            edge = tuple(sorted([vertex, new_vertex]))
            if edge in cluster.edges:
                neighbor_count += 1
                
        return neighbor_count == 2**self.d

    def recurence_step(self, step):
        """Grow clusters by one vertex"""
        if step not in self.size_to_clusters_boundaries_pair:
            return
            
        group = self.size_to_clusters_boundaries_pair[step]
        new_group = set()
        
        for (cluster_tuple, boundary) in group:
            cluster = self._tuple_to_graph(cluster_tuple)
            
            # Try to add new vertices to each boundary vertex
            for vert in boundary:
                for e in cluster.directions:
                    new_vertex = tuple(vert[i] + e[i] for i in range(self.d))
                    
                    # Skip if vertex already in cluster
                    if new_vertex in cluster.vertices:
                        continue
                    
                    # Create new cluster
                    new_cluster = ZDSubgraph(d=self.d)
                    # Copy all vertices and edges
                    for v in cluster.vertices:
                        new_cluster.add_vertex(v)
                    for edge in cluster.edges:
                        new_cluster.add_edge(edge[0], edge[1])
                    
                    # Add new vertex and edge
                    new_cluster.add_vertex(new_vertex)
                    new_cluster.add_edge(vert, new_vertex)
                    
                    # Calculate new boundary vertex
                    new_boundary = set(boundary.copy())
                    new_boundary.add(new_vertex)
                    

                    # Convert to hashable representation
                    new_cluster_tuple = self._graph_to_tuple(new_cluster)
                    if self.is_a_inner_vertex(vert, new_cluster_tuple):
                        new_boundary.remove(vert)
                    
                    
                    # Normalize the cluster
                    new_cluster, norm_boundary = normalize_origin(new_cluster, new_boundary, self.d)
                    
                    # Convert to hashable representation
                    new_boundary_frozen = frozenset(norm_boundary)
                    new_cluster_tuple = self._graph_to_tuple(new_cluster)

                    new_group.add((new_cluster_tuple, new_boundary_frozen))
        
        if new_group:
            self.size_to_clusters_boundaries_pair[step + 1] = new_group

    def run(self):
        """Run the cluster growth algorithm"""
        size_to_numb_clusters = dict()
        size_to_numb_clusters[1]=1
        for step in tqdm(range(1, self.s), leave=True):
            self.recurence_step(step)
            size_to_numb_clusters[step+1] = len(self.size_to_clusters_boundaries_pair.get(step+1, []))
        
        for size in range(1,self.s+1):
            print(f"Step {size}: {size_to_numb_clusters[size]} clusters")


        # Plot the final clusters

        if self.plot:
            final_clusters_boundary_pair = []
            for cluster_tuple, boundary in self.size_to_clusters_boundaries_pair[self.s]:
                cluster = self._tuple_to_graph(cluster_tuple)
                final_clusters_boundary_pair.append((cluster,boundary))
            
            plot_subgraphs_table(
                final_clusters_boundary_pair,
                max_plots=self.max_plots,
                title=f"Clusters of size {self.s}"
            )

    def get_cluster_counts(self):
        """Return the number of clusters of each size"""
        counts = {}
        for size, clusters in self.size_to_clusters_boundaries_pair.items():
            counts[size] = len(clusters)
        return counts
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialization Arguments")
    cluster_number_density.add_arguments(parser)
    args = parser.parse_args() # Use empty list to avoid command line parsing

    c = cluster_number_density(vars(args))
    c.run()