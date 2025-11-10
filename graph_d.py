import numpy as np

class ZDSubgraph:
    def __init__(self, d=2):
        """
        Initialize a subgraph of Z^d
        
        Args:
            d: dimension of the integer lattice
        """
        self.d = d
        self.vertices = set()
        self.edges = set()
        self.uf = UnionFindZD(d)
        self.directions = []
        for i in range(d):
            e = [0]*d
            e[i]=1
            neg_e = [0]*d
            neg_e[i] = -1
            self.directions.append(tuple(e))
            self.directions.append(tuple(neg_e))
        
    def add_vertex(self, coord):
        """Add a vertex to the subgraph"""

        # Check if is a vertice of Z^d
        if len(coord) != self.d:
            raise ValueError(f"Invalid vertice dimension {len(coord)} !=  d = {self.d}")
        
        # Check if has integers coordinates
        for c in coord:
            assert isinstance(c,int)

        if coord not in self.vertices:
            self.vertices.add(coord)
            self.uf.add_vertex(coord)
    
    def add_edge(self, coord1, coord2):
        """Add an edge between two vertices"""
        # Verify vertices exist and edge is valid
        if coord1 not in self.vertices:
            self.add_vertex(coord1)
        if coord2 not in self.vertices:
            self.add_vertex(coord2)
        
        # Check if it's a valid Z^d edge (differ in exactly one coordinate by 1)
        diff = tuple(abs(a - b) for a, b in zip(coord1, coord2))
        if sum(diff) != 1 or max(diff) != 1:
            raise ValueError(f"Invalid edge in Z^{self.d}: {coord1} - {coord2}")
        
        edge = tuple(sorted([coord1, coord2]))
        if edge not in self.edges:
            self.edges.add(edge)
            self.uf.union(coord1, coord2)
    
    def get_boundary(self):
        """
        Count the boundary of the graph in Z^d
        Returns edges between vertices in the graph and vertices not in the graph
        """
        boundary_edges = set()
        
        for vertex in self.vertices:
            # Check all adjacent positions in Z^d
            for dim in range(self.d):
                for delta in [-1, 1]:
                    neighbor = list(vertex)
                    neighbor[dim] += delta
                    neighbor = tuple(neighbor)
                    testing_edge = tuple(sorted([vertex,neighbor]))
                    
                    if testing_edge not in self.edges:
                        boundary_edge = tuple(sorted([vertex, neighbor]))
                        boundary_edges.add(boundary_edge)
        
        return boundary_edges
    
    def boundary_size(self):
        """Return the number of boundary edges"""
        return len(self.get_boundary())
    
    def connected_components(self):
        """Return the number of connected components"""
        return self.uf.num_components()
    def get_components(self):
        """Return a list of connected components sorted by size"""
        components = []
        for value in self.uf.get_component_sizes().values():
            components.append(value)
        return sorted(components)
    
    def is_connected(self, coord1, coord2):
        """Check if two vertices are connected"""
        return self.uf.find(coord1) == self.uf.find(coord2)

class UnionFindZD:
    def __init__(self, d=2):
        """
        Union-Find for Z^d coordinates
        
        Args:
            d: dimension of coordinates
        """
        self.d = d
        self.parent = {}
        self.size = {}
        self.coord_to_id = {}
        self.id_to_coord = {}
        self.next_id = 0
    
    def _get_id(self, coord):
        """Convert coordinate to unique integer ID"""
        if coord not in self.coord_to_id:
            self.coord_to_id[coord] = self.next_id
            self.id_to_coord[self.next_id] = coord
            self.next_id += 1
        return self.coord_to_id[coord]
    
    def _get_coord(self, id_val):
        """Convert ID back to coordinate"""
        return self.id_to_coord[id_val]
    
    def add_vertex(self, coord):
        """Add a new vertex to the Union-Find structure"""
        id_val = self._get_id(coord)
        if id_val not in self.parent:
            self.parent[id_val] = id_val
            self.size[id_val] = 1
    
    def find(self, coord):
        """Find root with path compression"""
        id_val = self._get_id(coord)
        
        # Initialize tracking
        current = id_val
        path = []
        
        # Find root
        while self.parent[current] != current:
            path.append(current)
            current = self.parent[current]
        
        root = current
        
        # Path compression
        for node in path:
            self.parent[node] = root
            
        return root
    
    def union(self, coord1, coord2):
        """Union by size"""
        root1 = self.find(coord1)
        root2 = self.find(coord2)
        
        if root1 != root2:
            # Union by size
            if self.size[root1] < self.size[root2]:
                self.parent[root1] = root2
                self.size[root2] += self.size[root1]
            else:
                self.parent[root2] = root1
                self.size[root1] += self.size[root2]
    
    def num_components(self):
        """Return number of connected components"""
        roots = set()
        for id_val in self.parent:
            roots.add(self.find(self._get_coord(id_val)))
        return len(roots)
    
    def get_component_sizes(self):
        """Return sizes of all connected components"""
        component_sizes = dict()
        for id_val in self.parent:
            root = self.find(self._get_coord(id_val))
            component_sizes[root] = self.size[root]
        return dict(component_sizes)

# Example usage and demonstration
def example_usage():
    # Create a subgraph in Z^2
    G = ZDSubgraph(d=2)
    
    # Add some vertices
    vertices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
    for v in vertices:
        G.add_vertex(v)
    
    # Add edges to form a square and an isolated vertex
    G.add_edge((0, 0), (0, 1))
    G.add_edge((0, 0), (1, 0))
    G.add_edge((0, 1), (1, 1))
    G.add_edge((1, 0), (1, 1))
    
    print(f"Number of vertices: {len(G.vertices)}")
    print(f"Number of edges: {len(G.edges)}")
    print(f"Number of connected components: {G.connected_components()}")
    print(f"Boundary size: {G.boundary_size()}")
    
    # Check connectivity
    print(f"(0,0) connected to (1,1): {G.is_connected((0,0), (1,1))}")
    print(f"(0,0) connected to (2,2): {G.is_connected((0,0), (2,2))}")
    
    # Show boundary edges
    boundary = G.get_boundary()
    print(f"Boundary edges: {sorted(boundary)}")

if __name__ == "__main__":
    example_usage()