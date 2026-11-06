"""
Constructs a graph with 32 vertices: 8 disjoint copies of K4.
- No 5-cycles: Max cycle length in K4 is 4.
- No independent set of 9: Independence number of K4 is 1, so total alpha is 8.
- n = 32.
"""

def construct():
    # 8 blocks of 4 vertices each
    num_blocks = 8
    block_size = 4
    n = num_blocks * block_size
    
    edges = []
    for b in range(num_blocks):
        start = b * block_size
        for i in range(block_size):
            for j in range(i + 1, block_size):
                u = start + i
                v = start + j
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
