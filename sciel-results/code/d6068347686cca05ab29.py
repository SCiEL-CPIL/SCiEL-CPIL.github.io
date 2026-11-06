"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 16
    num_edges = n * (n - 1) // 2
    colors = [0] * num_edges
    
    # Helper to get index of edge (i, j) in lexicographic order
    def edge_index(i, j):
        if i > j:
            i, j = j, i
        # Number of edges starting with vertices < i
        # For vertex 0: 15 edges
        # For vertex 1: 14 edges
        # ...
        # For vertex i: n - 1 - i edges
        # Sum_{k=0}^{i-1} (n - 1 - k) = i*(n-1) - i*(i-1)/2
        # Plus offset j - i - 1
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Construct a proper edge coloring using a 1-factorization of K16.
    # This uses n - 1 = 15 colors.
    # Vertices 0..14 form a cycle, 15 is the center (infinity).
    for k in range(15):
        # Edge connecting infinity (15) to k
        u, v = 15, k
        idx = edge_index(u, v)
        colors[idx] = k
        
        # Edges in the matching on vertices 0..14
        for i in range(1, 8):
            u = (k - i) % 15
            v = (k + i) % 15
            idx = edge_index(u, v)
            colors[idx] = k
            
    return colors
# EVOLVE-BLOCK-END
