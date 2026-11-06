import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n = 16
    # Grötzsch graph edges (11 vertices: 0..10)
    # Cycle 0-1-2-3-4-0
    # 5 connects to 1, 4
    # 6 connects to 2, 0
    # 7 connects to 3, 1
    # 8 connects to 4, 2
    # 9 connects to 0, 3
    # 10 connects to 5, 6, 7, 8, 9
    core_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 4),
        (6, 2), (6, 0),
        (7, 3), (7, 1),
        (8, 4), (8, 2),
        (9, 0), (9, 3),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    # All possible pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def run_greedy(seed):
        random.seed(seed)
        # Initialize adjacency as bitsets
        adj = [0] * n
        for u, v in core_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # List of non-core edges
        core_set = set(core_edges)
        # Note: core_edges are tuples (u,v) with u<v? 
        # In definition above, yes.
        # Let's ensure all_pairs are u<v
        non_core_edges = [e for e in all_pairs if e not in core_set]
        
        random.shuffle(non_core_edges)
        
        edge_count = len(core_edges)
        
        for u, v in non_core_edges:
            # Check if adding (u,v) creates a triangle
            # A triangle is formed if there exists w such that (u,w) and (v,w) are in adj
            # This is equivalent to (adj[u] & adj[v]) != 0
            if (adj[u] & adj[v]) == 0:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
                
        # Convert to list of edges
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    best_edges = []
    best_count = 0
    
    # Run multiple random restarts
    # Estimate: 120s / (time per restart)
    # Time per restart is very small (120 edges, simple bit ops)
    # Let's try 5000 restarts
    for seed in range(5000):
        if time.time() > deadline:
            break
        edges = run_greedy(seed)
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = edges
            
    # Optional: Local Search on the best result
    # Try to improve by swapping edges?
    # Since we are maximizing edges, and we already filled greedily, 
    # local search might find a configuration with more edges 
    # if the greedy choice blocked multiple other edges.
    
    # A simple 1-1 swap doesn't increase count.
    # We need a "2-for-1" or similar move? 
    # Or just accept the greedy result as it is likely near optimal for this size.
    
    return best_edges
