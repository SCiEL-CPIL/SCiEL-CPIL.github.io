"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 16}

def construct():
    """
    Constructs the complete bipartite graph K_{8,8} on vertices 0..15.
    
    Reasoning:
    1. The graph must be triangle-free. If it contained a triangle (u, v, w), 
       removing any edge (e.g., u-v) would leave the graph connected with 
       dist(u, v) = 2 (via w), so the diameter would remain 2. This violates 
       the condition that deleting each edge must increase diameter or disconnect.
    2. The maximum number of edges in a triangle-free graph on n vertices is 
       given by Turan's theorem: floor(n^2 / 4). For n=16, this is 64.
    3. The complete bipartite graph K_{8,8} has 64 edges, is triangle-free, 
       and has diameter 2 (since both partitions have size > 1).
    4. In K_{8,8}, removing any edge (u, v) increases the distance between u 
       and v to 3 (path u -> w -> x -> v where w is in the other partition 
       of u and x is in the other partition of v, specifically u->v'->u'->v 
       is not direct, it's u->b'->a'->v? No, u in A, v in B. 
       Path: u -> b' (in B, b'!=v) -> a' (in A, a'!=u) -> v. 
       Wait, b' is in B, a' is in A. 
       u is in A. Neighbors of u are all in B. 
       v is in B. Neighbors of v are all in A.
       If we remove (u,v), can we go u->...->v in 2 steps? 
       u->b' (b' in B). b'->v? No, b' and v are both in B, no edge.
       So dist(u,v) becomes 3.
       Thus diameter becomes 3.
       So K_{8,8} is a valid solution with the maximum possible edges.
    """
    n = P['n']
    mid = n // 2
    set_a = list(range(mid))
    set_b = list(range(mid, n))
    
    edges = []
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
