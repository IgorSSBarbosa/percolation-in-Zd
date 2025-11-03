from graph_d import ZDSubgraph
from Z2_plot import plot_subgraphs_table
from tqdm import tqdm
from itertools import chain, combinations


class ClusterGenerator:
    def __init__(self, s, d=2):
        self.s = s
        self.d = d
        
        # Initialize with origin
        graph = ZDSubgraph(d=d)
        origin = tuple([0] * d)
        graph.add_vertex(origin)
        
        self.size_to_clusters_boundaries_pair = dict()
        boundary = self._calculate_boundary(graph)
        
        initial_pair = set()
        initial_pair.add((self._graph_to_tuple(graph), frozenset(boundary)))
        self.size_to_clusters_boundaries_pair[1] = initial_pair
        print(f"Step 1: {len(self.size_to_clusters_boundaries_pair[1])} clusters")

    def _calculate_boundary(self, graph):
        """Calculate boundary vertices (vertices that can be extended)"""
        boundary = set()
        for vertex in graph.vertices:
            for direction in graph.directions:
                neighbor = tuple(vertex[i] + direction[i] for i in range(self.d))
                if neighbor not in graph.vertices:
                    boundary.add(vertex)
                    break  # No need to check other directions once we know it's boundary
        return boundary

    def _graph_to_tuple(self, graph):
        """Convert graph to hashable tuple"""
        vertices = tuple(sorted(graph.vertices))
        edges = tuple(sorted(tuple(sorted(edge)) for edge in graph.edges))
        return (vertices, edges)

    def _tuple_to_graph(self, graph_tuple):
        """Convert tuple back to graph"""
        vertices, edges = graph_tuple
        graph = ZDSubgraph(d=self.d)
        for vertex in vertices:
            graph.add_vertex(vertex)
        for edge in edges:
            graph.add_edge(edge[0], edge[1])
        return graph

    def is_inner_vertex(self, vertex, graph_tuple):
        """Check if vertex is inner (all neighbors are in graph)"""
        graph = self._tuple_to_graph(graph_tuple)
        for direction in graph.directions:
            neighbor = tuple(vertex[i] + direction[i] for i in range(self.d))
            if neighbor not in graph.vertices:
                return False
        return True

    def get_possible_edges(self, graph):
        """Get all possible edges that can be added to form cycles"""
        possible_edges = set()
        vertices = list(graph.vertices)
        
        # Check all pairs of vertices that are adjacent in Z^d but not connected in graph
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                v1, v2 = vertices[i], vertices[j]
                dist = sum(abs(v1[k] - v2[k]) for k in range(self.d))
                if dist == 1:  # Adjacent in Z^d
                    edge = tuple(sorted([v1, v2]))
                    if edge not in graph.edges:
                        possible_edges.add(edge)
        
        return possible_edges

    def recurence_step(self, step):
        """Grow clusters by one vertex AND add possible cycles"""
            
        group = self.size_to_clusters_boundaries_pair[step]
        new_group = set()
        
        for (cluster_tuple, boundary) in group:
            cluster = self._tuple_to_graph(cluster_tuple)
            
            # PART 1: Add new vertices (tree growth)
            for vert in boundary:
                for direction in cluster.directions:
                    new_vertex = tuple(vert[i] + direction[i] for i in range(self.d))
                    
                    if new_vertex in cluster.vertices:
                        continue
                    
                    # Create new cluster with added vertex
                    new_cluster = self._add_vertex_to_graph(cluster, vert, new_vertex)
                    new_boundary = self._update_boundary(boundary, vert, new_vertex, new_cluster)
                    
                    # Normalize and store
                    new_cluster_norm, new_boundary_norm = normalize_origin_with_boundary(new_cluster, new_boundary)
                    new_group.add((self._graph_to_tuple(new_cluster_norm), frozenset(new_boundary_norm)))
            
        final_group = set()
        # PART 2: Add cycles (only if we have enough vertices)
        for (new_cluster_tuple, new_boundary) in new_group:
            # adding old graphs to final group
            final_group.add((new_cluster_tuple,new_boundary))
            new_cluster = self._tuple_to_graph(new_cluster_tuple)
            if len(new_cluster.vertices) >= 3:  # Need at least 3 vertices for a cycle
                possible_edges = self.get_possible_edges(new_cluster)
                power_set_edges = subsets(possible_edges)
                for subset_edges in power_set_edges:   
                    for edge in subset_edges:
                        v1, v2 = edge
                        
                        cycle_cluster = self._add_edge_to_graph(new_cluster, v1, v2)
                        cycle_boundary = self._update_boundary_for_cycle(new_boundary, v1, v2, cycle_cluster)
                        
                    # Normalize and store
                    cycle_cluster_norm, cycle_boundary_norm = normalize_origin_with_boundary(cycle_cluster, cycle_boundary)
                    final_group.add((self._graph_to_tuple(cycle_cluster_norm), frozenset(cycle_boundary_norm)))
        
        if final_group:
            self.size_to_clusters_boundaries_pair[step + 1] = final_group

    def _add_vertex_to_graph(self, graph, from_vertex, new_vertex):
        """Create a new graph with added vertex and edge"""
        new_graph = ZDSubgraph(d=self.d)
        # Copy all vertices
        for vertex in graph.vertices:
            new_graph.add_vertex(vertex)
        # Copy all edges
        for edge in graph.edges:
            new_graph.add_edge(edge[0], edge[1])
        # Add new vertex and edge
        new_graph.add_vertex(new_vertex)
        new_graph.add_edge(from_vertex, new_vertex)
        return new_graph

    def _add_edge_to_graph(self, graph, v1, v2):
        """Create a new graph with added edge"""
        new_graph = ZDSubgraph(d=self.d)
        # Copy all vertices
        for vertex in graph.vertices:
            new_graph.add_vertex(vertex)
        # Copy all edges
        for edge in graph.edges:
            new_graph.add_edge(edge[0], edge[1])
        # Add new edge
        new_graph.add_edge(v1, v2)
        return new_graph

    def _update_boundary(self, old_boundary, from_vertex, new_vertex, new_graph):
        """Update boundary after adding a new vertex"""
        new_boundary = set(old_boundary)
        
        # Check if from_vertex becomes inner
        if self.is_inner_vertex(from_vertex, self._graph_to_tuple(new_graph)):
            new_boundary.discard(from_vertex)
        
        # Add new_vertex to boundary (it might become inner later, but starts as boundary)
        new_boundary.add(new_vertex)
        
        return new_boundary

    def _update_boundary_for_cycle(self, old_boundary, v1, v2, new_graph):
        """Update boundary after adding a cycle edge"""
        new_boundary = set(old_boundary)
        
        # Check if v1 or v2 become inner vertices
        if self.is_inner_vertex(v1, self._graph_to_tuple(new_graph)):
            new_boundary.discard(v1)
        if self.is_inner_vertex(v2, self._graph_to_tuple(new_graph)):
            new_boundary.discard(v2)
            
        return new_boundary

    def run(self):
        """Generate all clusters up to size s"""
        for step in tqdm(range(1, self.s)):
            self.recurence_step(step)
            print(f"Step {step+1}: {len(self.size_to_clusters_boundaries_pair[step+1])} clusters")
        
        final_size = min(self.s, max(self.size_to_clusters_boundaries_pair.keys()))
        final_clusters = []
        
        for cluster_tuple, boundary in self.size_to_clusters_boundaries_pair.get(final_size, set()):
            cluster = self._tuple_to_graph(cluster_tuple)
            final_clusters.append((cluster,boundary))
        
        return final_clusters

    def get_statistics(self):
        """Get statistics about generated clusters"""
        stats = {}
        for size, clusters in self.size_to_clusters_boundaries_pair.items():
            stats[size] = {
                'count': len(clusters),
                'trees': 0,
                'with_cycles': 0
            }
            for cluster_tuple, _ in clusters:
                cluster = self._tuple_to_graph(cluster_tuple)
                # A graph is a tree if |E| = |V| - 1
                if len(cluster.edges) == len(cluster.vertices) - 1:
                    stats[size]['trees'] += 1
                else:
                    stats[size]['with_cycles'] += 1
        return stats

def subsets(input_set):
    """Returns all the nonempty subsets of a given set"""
    assert isinstance(input_set,set)
    power_set = set()
    s = list(input_set)
    for i in range(1,len(s)+1):
        for subset in combinations(s,i):
            power_set.add(subset)

    return power_set



# Keep the oigin as the vertex with minimun y, and between all vertex with minimun y the origin has minimum x
def normalize_origin_with_boundary(graph, boundary):
    """Normalize graph and adjust boundary accordingly"""
    if not graph.vertices:
        return graph, set()
    
    # Find translation vector
    min_y = min(y for (x, y) in graph.vertices)
    vertices_at_min_height = [(x, y) for (x, y) in graph.vertices if y == min_y]
    leftmost_x = min(x for (x, y) in vertices_at_min_height)
    
    translate_x, translate_y = -leftmost_x, -min_y
    
    # Create normalized graph
    normalized_graph = ZDSubgraph(d=graph.d)
    vertex_map = {}
    
    for vertex in graph.vertices:
        new_vertex = (vertex[0] + translate_x, vertex[1] + translate_y)
        vertex_map[vertex] = new_vertex
        normalized_graph.add_vertex(new_vertex)
    
    for edge in graph.edges:
        new_edge = (vertex_map[edge[0]], vertex_map[edge[1]])
        normalized_graph.add_edge(new_edge[0], new_edge[1])
    
    # Translate boundary
    normalized_boundary = set(vertex_map[vertex] for vertex in boundary)
    
    return normalized_graph, normalized_boundary


if __name__=="__main__":
    s=9
    d=2
    group_generator = ClusterGenerator(s,d)

    group = group_generator.run()
    plot_subgraphs_table(group, max_plots=500, title=f'ZD Clusters of size {s}')