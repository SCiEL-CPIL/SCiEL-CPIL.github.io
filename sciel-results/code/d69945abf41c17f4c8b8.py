import time

def construct():
    # We construct a graph consisting of 8 disjoint copies of K4.
    # This graph has n = 8 * 4 = 32 vertices.
    # It is C5-free because each component is a clique of size 4 (no cycles of length >= 5).
    # Its independence number is 8 (one vertex from each K4), so it has no independent set of size 9.
    
    n = 32
    num_blocks = 8
    block_size = 4
    
    edges = []
    for b in range(num_blocks):
        start = b * block_size
        # Generate all edges within this block of 4 vertices
        for i in range(block_size):
            for j in range(i + 1, block_size):
                u = start + i
                v = start + j
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
