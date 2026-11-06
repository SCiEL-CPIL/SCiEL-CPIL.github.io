"""Rewritten program with improved graph construction."""
# EVOLVE-BLOCK-START
def construct():
    n = 16
    # Partition vertices into two sets of 8
    v1 = list(range(8))
    v2 = list(range(8, 16))
    
    edges = []
    
    # Start with K_{8,8}
    for u in v1:
        for v in v2:
            edges.append([u, v])
            
    # Remove one edge (0, 8) to allow adding two intra-partition edges
    # while keeping total edge count at 65 (64 - 1 + 2 = 65)
    edges.remove([0, 8])
    
    # Add two intra-partition edges
    # This creates triangles with degree sums of 25
    edges.append([0, 1])
    edges.append([8, 9])
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
