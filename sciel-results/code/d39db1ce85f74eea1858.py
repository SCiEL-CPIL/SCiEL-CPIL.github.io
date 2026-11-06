import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    deadline = time.time() + 115.0

    def compute_books(A):
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        return A2

    # Initialize with a bipartite graph K_20,20 (400 edges) + 1 extra edge
    # This minimizes triangles initially (0 in bipartite, adding 1 edge creates small books)
    edges = set()
    A = np.zeros((n, n), dtype=np.int32)
    
    # Partition 0..39 into two sets of 20
    left = list(range(20))
    right = list(range(20, 40))
    
    # Add all edges between left and right
    for u in left:
        for v in right:
            edges.add((u, v))
            A[u, v] = 1
            A[v, u] = 1
            
    # We have 400 edges. Need 1 more.
    # Add a random edge within one partition to start
    u = random.choice(left)
    v = random.choice(left)
    while u == v:
        v = random.choice(left)
    new_edge = (min(u, v), max(u, v))
    edges.add(new_edge)
    A[u, v] = 1
    A[v, u] = 1

    A2 = compute_books(A)
    max_b = np.max(A2)
    best_edges = set(edges)
    best_max_b = max_b

    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.9999
    
    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() < deadline - 2:
        # Pick a random edge to remove
        rem_u, rem_v = random.choice(list(edges))
        
        # Pick a random non-edge to add
        # To be efficient, we can pick a random pair and check if it's in edges
        # Or maintain a list of non-edges. For n=40, 780 total edges, 401 in graph, 379 non-edges.
        # Random sampling is fine.
        added = False
        for _ in range(50): # Try a few times to find a valid non-edge
            add_u = random.randint(0, n - 1)
            add_v = random.randint(0, n - 1)
            if add_u == add_v:
                continue
            add_edge = (min(add_u, add_v), max(add_u, add_v))
            if add_edge not in edges:
                added = True
                break
        
        if not added:
            continue
            
        # Apply swap
        A[rem_u, rem_v] = 0
        A[rem_v, rem_u] = 0
        A[add_u, add_v] = 1
        A[add_v, add_u] = 1
        
        A2_new = compute_books(A)
        max_new = np.max(A2_new)
        
        # Calculate delta
        delta = max_new - max_b
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
            # Accept move
            edges.discard((rem_u, rem_v))
            edges.add(add_edge)
            A2 = A2_new
            max_b = max_new
            
            if max_b < best_max_b:
                best_max_b = max_b
                best_edges = set(edges)
                T = max(T_min, T * 0.95) # Cool down faster on improvement? Or standard cooling
            else:
                T = max(T_min, T * cooling)
        else:
            # Revert
            A[rem_u, rem_v] = 1
            A[rem_v, rem_u] = 1
            A[add_u, add_v] = 0
            A[add_v, add_u] = 0
            T = max(T_min, T * cooling)

    return [[u, v] for u, v in best_edges]
