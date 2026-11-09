import time

def construct():
    """
    Constructs a graph consisting of a path of 8 K4 cliques connected by single edges.
    n = 32.
    No 5-cycle: The graph is formed by connecting K4s (which have max cycle length 4) via bridges. 
    Bridges cannot be part of a simple cycle. Thus, all cycles are within the K4s (C3, C4).
    No independent set of 9: Each K4 contributes at most 1 vertex to an independent set 
    (since it's a clique). The connecting edges prevent picking adjacent bridge vertices 
    from different cliques simultaneously, but we can pick one vertex from each K4.
    Total alpha = 8 (one from each K4). 8 < 9.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin well within 120s limit
    
    n = 32
    edges = []
    
    # 8 blocks of 4 vertices
    num_blocks = 8
    block_size = 4
    
    for b in range(num_blocks):
        if time.time() > deadline:
            break
        base = b * block_size
        # Add all edges within the K4 block
        for i in range(block_size):
            if time.time() > deadline:
                break
            for j in range(i + 1, block_size):
                edges.append([base + i, base + j])
        
        # Connect this block to the next block with a single edge
        if b < num_blocks - 1:
            # Connect vertex 0 of block b to vertex 0 of block b+1
            next_base = (b + 1) * block_size
            edges.append([base + 0, next_base + 0])

    return {'n': n, 'edges': edges}
