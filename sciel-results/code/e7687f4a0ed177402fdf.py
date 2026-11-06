"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Base graph G on 7 vertices (0..6)
    # C5: 0-1-2-3-4-0
    # 5 connected to 0, 2
    # 6 connected to 1, 3
    # This graph is triangle-free and 3-chromatic (contains C5)
    base_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [5, 0], [5, 2],
        [6, 1], [6, 3]
    ]
    
    n_base = 7
    n_mycielski = 2 * n_base + 1  # 15 vertices (0..14)
    
    edges = []
    
    # 1. Add edges of G (vertices 0..6)
    for u, v in base_edges:
        edges.append([u, v])
        
    # 2. Add edges between U (7..13) and V (0..6)
    # u_i (7+i) is connected to v_j if v_i is connected to v_j in G
    for u, v in base_edges:
        # u_i connected to v_j
        edges.append([7 + u, v])
        # v_i connected to u_j (symmetric)
        edges.append([7 + v, u])
        
    # 3. Add edges from Z (14) to all U (7..13)
    for i in range(n_base):
        edges.append([14, 7 + i])
        
    # 4. Vertex 15 is isolated
    
    return edges
# EVOLVE-BLOCK-END
