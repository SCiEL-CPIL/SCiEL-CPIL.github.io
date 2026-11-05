import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 40
    m = 401
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: idx for idx, e in enumerate(all_edges)}
    
    # Precompute neighbors for each edge (common neighbors)
    # For speed, we use bitsets (integers) for adjacency
    # adj[i] is an integer where bit j is 1 if (i, j) is an edge
    
    def compute_books(adj):
        max_book = 0
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    # Count common neighbors
                    common = adj[i] & adj[j]
                    # Popcount
                    cnt = bin(common).count('1')
                    if cnt > max_book:
                        max_book = cnt
        return max_book

    # Strategy: 
    # 1. Initialize with a structured graph (e.g., based on quadratic residues or random)
    # 2. Use Simulated Annealing with edge swaps
    #    - Pick a random edge (u,v) and a random non-edge (x,y)
    #    - Swap them (add x-y, remove u-v) if it doesn't violate constraints
    #    - Actually, simpler: Pick two edges (u,v) and (x,y), swap to (u,x) and (v,y) or similar?
    #    - Better: Pick an edge (u,v) and a non-edge (x,y). Remove (u,v), add (x,y).
    #    - This keeps m constant.
    
    # Let's use a bitset representation for fast updates.
    # adj is a list of ints.
    
    # Initial state: Random graph
    random.seed(42)
    initial_edges = random.sample(all_edges, m)
    adj = [0] * n
    for u, v in initial_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    current_score = compute_books(adj)
    best_score = current_score
    best_adj = adj[:]
    
    # To allow efficient local search, we need to update score quickly.
    # Full recomputation is O(m * n/word_size) ~ 401 * 1 = fast.
    # With bitsets, computing one edge's book is O(1) (popcount of AND).
    # Computing max book is O(m).
    # 401 edges * 40 bits = very fast.
    # We can afford to recompute the whole score for every move.
    
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline:
        # Pick a random edge to remove
        u, v = random.choice(all_edges)
        if not ((adj[u] >> v) & 1):
            continue # Not an edge
        
        # Pick a random non-edge to add
        x, y = random.choice(all_edges)
        if (adj[x] >> y) & 1:
            continue # Already an edge
        if {x, y} == {u, v}:
            continue
        
        # Perform swap
        # Remove (u, v)
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        # Add (x, y)
        adj[x] ^= (1 << y)
        adj[y] ^= (1 << x)
        
        new_score = compute_books(adj)
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta < 0 or (random.random() < np.exp(-delta / T)):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = adj[:]
        else:
            # Revert
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            adj[x] ^= (1 << y)
            adj[y] ^= (1 << x)
            
        T *= cooling
        steps += 1
        
        # Restart cooling if stuck?
        if steps % 1000 == 0:
            T = max(T, 0.1)
            
    # Reconstruct edges from best_adj
    best_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                best_edges.append([i, j])
                
    return best_edges

