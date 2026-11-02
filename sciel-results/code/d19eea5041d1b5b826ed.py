import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Target: 4-regular, girth >= 10
    # Try to find a graph with n < 2048
    # Start with a moderate size, e.g., 800
    # If not found, try smaller? No, smaller is harder.
    # Try larger? No, we want minimal n.
    
    # The LUW graph is 2048.
    # We will try to construct a graph with n=800, 900, etc.
    
    best_n = 2048
    best_edges = None
    
    # Fallback: Generate LUW graph if search fails
    def get_luw():
        q = k
        d = 5 # for girth >= 10
        # ... LUW construction ...
        # For brevity, I will implement a simple random search first.
        pass

    # Randomized Greedy Search
    for N in [1000, 900, 800, 700]:
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(20): # 20 attempts per N
            if time.time() > deadline:
                break
                
            # Generate random 4-regular graph
            edges = generate_random_regular_graph(N, k)
            if edges is None:
                continue
                
            # Check girth
            girth = compute_girth(N, edges)
            
            if girth >= g:
                best_n = N
                best_edges = edges
                found = True
                break
            
            # Try to improve girth by edge switching
            improved = try_improve_girth(N, edges, k, g, deadline)
            if improved:
                girth = compute_girth(N, edges)
                if girth >= g:
                    best_n = N
                    best_edges = edges
                    found = True
                    break
        
        if found:
            break

    if best_edges is not None:
        return best_n, best_edges
    else:
        # Fallback to LUW
        return construct_luw(k, g)

def generate_random_regular_graph(n, k):
    # Simple method: generate random permutations and combine
    # Or use networkx if available? No, only stdlib/NumPy.
    # Use a simple random matching approach.
    # This is tricky for k=4.
    # We can use the configuration model.
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    edges = set()
    for i in range(0, n*k, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        if (u, v) in edges or (v, u) in edges:
            return None
        edges.add((u, v))
    
    # Check regularity (should be)
    return list(edges)

def compute_girth(n, edges):
    # BFS from each vertex to find shortest cycle
    # Too slow for n=1000 in Python.
    # Use NumPy?
    # Or just return a large number if we assume it's good?
    # No, must be certified.
    # For n=1000, BFS is O(n*m) = 1000 * 2000 = 2M ops.
    # 1000 BFS = 2B ops. Too slow.
    # We need a faster girth check or smaller n.
    pass

def construct_luw(k, g):
    # Original LUW code
    pass
