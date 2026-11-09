<<<<<<< SEARCH
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial graph: K_{3,13}
    # U = {0, 1, 2}, V = {3, ..., 15}
    edges = set()
    for u in range(3):
        for v in range(3, 16):
            edges.add((u, v))
    
    # Precompute bitmask for each vertex: bit j is set if (i,j) in edges
    # Use a 16-bit integer per vertex
    adj_mask = [0] * 16
    for (u, v) in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
=======
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial graph: K_{4,12} plus a few intra-edges to boost density
    # U = {0, 1, 2, 3}, V = {4, ..., 15}
    edges = set()
    for u in range(4):
        for v in range(4, 16):
            edges.add((u, v))
    
    # Add some edges within U and within V carefully
    # Within U: add edges (0,1), (2,3) - these won't create Q3 immediately
    edges.add((0, 1))
    edges.add((2, 3))
    
    # Precompute bitmask for each vertex: bit j is set if (i,j) in edges
    # Use a 16-bit integer per vertex
    adj_mask = [0] * 16
    for (u, v) in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
>>>>>>> REPLACE

