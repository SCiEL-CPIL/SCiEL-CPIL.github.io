import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Baseline: Weight 2 vectors
    # These are always valid and provide a safe fallback
    baseline_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                baseline_vectors.append(v)
    baseline_count = len(baseline_vectors)
    
    # Try to find a better solution using Weight 3 vectors
    # Generate all weight 3 vectors
    weight3_vectors = []
    for combo in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, val in zip(combo, signs):
                v[idx] = val
            weight3_vectors.append(v)
    
    n_vecs = len(weight3_vectors)
    
    # Precompute conflict graph
    # Conflict if dot product > 1
    # We store conflicts as a list of bitmasks or just a matrix for small N
    # N=1320 is small enough for a dense matrix if we use numpy, but bitsets are faster for MIS heuristics
    # Let's use a list of lists of indices for adjacency
    
    # To speed up, we can use a greedy approach
    # 1. Calculate degrees
    # 2. Pick node with max degree (or min degree for independent set? No, usually max degree in greedy coloring, but for MIS we want to pick nodes that conflict with many others to prune? 
    #    Actually, standard greedy for MIS: pick a node, remove it and its neighbors. Repeat.
    #    Heuristic: Pick node with highest degree to maximize removal? Or lowest to preserve options?
    #    Max-min heuristic mentioned in prompt: pick node with max (min-distance to others)?
    #    Let's try a simple random restart greedy.
    
    best_set = []
    
    # Simple greedy with random restarts
    # We will run for a few seconds
    # If we find a set larger than baseline, we return it.
    
    # Optimization: Represent vectors as tuples for hashing? Or just indices.
    # Let's stick to indices into weight3_vectors.
    
    # Precompute dot products? 1320^2 / 2 ~ 870k operations. Fast.
    # dot[i][j] = weight3_vectors[i] . weight3_vectors[j]
    # Conflict if dot > 1.
    
    # We can just compute on the fly if we are careful, but precomputing adjacency list is better.
    
    # Build adjacency list
    adj = [set() for _ in range(n_vecs)]
    
    # Vectorized dot product calculation
    V = np.array(weight3_vectors, dtype=np.int8)
    # Compute all pairwise dot products
    # This might be memory heavy if done as full matrix, but 1320x1320 is fine.
    dots = V @ V.T
    
    for i in range(n_vecs):
        for j in range(i + 1, n_vecs):
            if dots[i, j] > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy MIS
    # Strategy: Randomly shuffle, then pick first available, remove neighbors.
    # To improve, sort by degree descending before picking?
    
    def greedy_mis(order):
        remaining = set(range(n_vecs))
        selected = []
        # We need to efficiently remove neighbors
        # Using a set for remaining is O(1) removal, but iterating is O(N)
        
        # Better: maintain a list of active nodes.
        # When we pick u, we remove u and adj[u] from active.
        
        active = set(range(n_vecs))
        
        for u in order:
            if u not in active:
                continue
            selected.append(u)
            # Remove u and its neighbors from active
            active.remove(u)
            for v in adj[u]:
                if v in active:
                    active.remove(v)
            
            if time.time() > deadline - 10:
                break
                
        return selected

    # Run several restarts
    best_count = 0
    best_indices = []
    
    # Try a few heuristics
    # 1. Random order
    # 2. Degree descending order
    
    # Calculate degrees
    degrees = [len(adj[i]) for i in range(n_vecs)]
    
    # Heuristic 1: Sort by degree descending
    order_deg = sorted(range(n_vecs), key=lambda x: -degrees[x])
    
    if time.time() < deadline - 20:
        res = greedy_mis(order_deg)
        if len(res) > best_count:
            best_count = len(res)
            best_indices = res
    
    # Heuristic 2: Random restarts
    num_restarts = 0
    while time.time() < deadline - 20:
        random.shuffle(order_deg) # Reuse the list, shuffle it
        # Actually, let's create a new random order
        order_rand = list(range(n_vecs))
        random.shuffle(order_rand)
        
        res = greedy_mis(order_rand)
        if len(res) > best_count:
            best_count = len(res)
            best_indices = res
        num_restarts += 1
        
        # If we've done a lot, maybe break
        if num_restarts > 1000:
            break

    # Compare with baseline
    if best_count > baseline_count:
        result_vectors = [weight3_vectors[i] for i in best_indices]
        return np.array(result_vectors, dtype=np.int64)
    else:
        return np.array(baseline_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
