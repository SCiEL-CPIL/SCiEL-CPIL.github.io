"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Use the Paley graph on 17 vertices (K4-free) plus an isolated vertex 17.
    # This structure is expected to have a smaller largest triangle-free induced
    # subgraph than the complete tripartite K6,6,6 (which has size 12).
    
    n = 18
    # Quadratic residues modulo 17
    # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=8, 6^2=2, 7^2=15, 8^2=13
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    
    edges = []
    # Generate edges for vertices 0..16
    for u in range(17):
        for v in range(u + 1, 17):
            diff = (v - u) % 17
            if diff in residues:
                edges.append([u, v])
    
    # Vertex 17 is isolated, so no edges involving 17.
    # The graph is K4-free because P(17) is K4-free and the isolated vertex
    # cannot be part of any K4.
    
    return edges
# EVOLVE-BLOCK-END
