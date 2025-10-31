from graph_d import ZDSubgraph


def normalize_origin(G, boundary, d):
    """
    Transform graph G so that:
    1. The origin is always the vertex with minimum height (y-coordinate)
    2. Among all vertices with height 0, the origin is the leftmost vertex
       (all other vertices of form (x,0) must have x > 0)
    
    Returns a new ZDSubgraph with the transformed coordinates
    """
    assert isinstance(G, type(ZDSubgraph(d)))
    
    # Find current minimum y-coordinate and leftmost vertex at that height
    min_y = min(y for (x, y) in G.vertices)
    vertices_at_min_height = [(x, y) for (x, y) in G.vertices if y == min_y]
    leftmost_x = min(x for (x, y) in vertices_at_min_height)
    
    # Calculate translation needed
    translate_x = -leftmost_x
    translate_y = -min_y
    
    # Create new graph
    G_normalized = ZDSubgraph(d=G.d)
    boundary_normalized = set()
    
    # Add translated vertices
    vertex_map = {}
    for vertex in G.vertices:
        new_vertex = tuple(v + t for v, t in zip(vertex, (translate_x, translate_y)))
        vertex_map[vertex] = new_vertex
        G_normalized.add_vertex(new_vertex)
        if vertex in boundary:
            boundary_normalized.add(new_vertex)
    
    # Add translated edges
    for edge in G.edges:
        new_edge = (vertex_map[edge[0]], vertex_map[edge[1]])
        G_normalized.add_edge(new_edge[0], new_edge[1])
    
    return (G_normalized,boundary_normalized)


# Test function to demonstrate the normalization
def test_normalization():
    """Test the origin normalization function"""
    
    # Create test graphs
    test_cases = [
        # Case 1: Graph already normalized
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        
        # Case 2: Graph needs vertical translation
        [(2, 3), (3, 3), (2, 4), (3, 4)],
        
        # Case 3: Graph needs horizontal translation
        [(-2, 0), (-1, 0), (-2, 1), (-1, 1)],
        
        # Case 4: Graph needs both translations
        [(-3, 5), (-2, 5), (-3, 6), (-2, 6)],
        
        # Case 5: Multiple vertices at minimum height
        [(-1, 2), (0, 2), (1, 2), (0, 3)]
    ]
    
    for i, vertices in enumerate(test_cases):
        print(f"\n=== Test Case {i+1} ===")
        
        # Create original graph
        G_original = ZDSubgraph(d=2)
        for v in vertices:
            G_original.add_vertex(v)
        
        # Add some edges
        for j in range(len(vertices)-1):
            try:
                G_original.add_edge(vertices[j], vertices[j+1])
            except:
                print(f"invalid edge ({vertices[j]},{vertices[j+1]})")
        
        empty_boundary = set()

        # Normalize
        G_normalized, boundary = normalize_origin(G_original, empty_boundary, d=2)
        print(f"Normalized graph:")
        print(f"  Vertices: {sorted(G_normalized.vertices)}")
        
        
        # # Verify properties
        assert (0, 0) in G_normalized.vertices, f"Test {i+1} failed: (0,0) should be in vertices"
        
        # Verify all vertices at height 0 have x >= 0
        vertices_at_height_0 = [(x, y) for (x, y) in G_normalized.vertices if y == 0]
        for x, y in vertices_at_height_0:
            assert x >= 0, f"Test {i+1} failed: vertex ({x},{y}) at height 0 has x < 0"
        
        print(f"  ✓ Test {i+1} passed!")


if __name__ == "__main__":
    test_normalization()