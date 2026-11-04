import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330
    
    # Convert supports to sets for fast intersection? Or just use bitmasks.
    # Bitmasks are fast for intersection.
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    # 2. Build conflict graph
    # Conflict if intersection size is 3
    # adj[i] is a set of indices j that conflict with i
    adj = [set() for _ in range(N_S)]
    
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter == 3:
                adj[i].add(j)
                adj[j].add(i)

    # 3. Find Maximum Independent Set using Greedy Randomized Restart
    best_size = 0
    best_set = set()
    
    # Precompute random priorities? No, shuffle order each time.
    rng = np.random.RandomState(42)
    
    # Number of restarts: try to do as many as possible
    # Each restart is O(N^2) in worst case if we check adjacency naively, 
    # but with a "selected" set and checking against it, it's O(N * deg).
    # N=330, deg=28. Very fast.
    
    num_restarts = 5000 # High number, but each is fast
    
    for r in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        # Random order
        order = rng.permutation(N_S)
        
        current_set = set()
        
        for v in order:
            # Check if v conflicts with any in current_set
            # If yes, skip. If no, add.
            # Optimization: if v is in current_set, skip (not possible here as order is unique)
            # Check neighbors of v
            is_conflict = False
            for u in adj[v]:
                if u in current_set:
                    is_conflict = True
                    break
            
            if not is_conflict:
                current_set.add(v)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # Optional: Local search improvement on current_set
        # Try to add vertices that were skipped?
        # Or try 1-swap: remove u, add v if v was blocked by u and no other conflicts.
        # Given the time budget, pure greedy restarts are likely sufficient and robust.
        # Let's add a simple local search step to polish.
        
        # Simple Local Search: 
        # While improvement possible:
        #   For each v not in set:
        #     Let N_conflict(v) be neighbors of v in current_set
        #     If len(N_conflict(v)) == 1, say {u}, and u has no other "critical" neighbors?
        #     Actually, simpler: 
        #     Try removing each u in current_set.
        #     If removing u allows adding k > 1 vertices, do it.
        #     This is complex. Let's stick to pure greedy restarts first.
        # The graph is sparse, greedy is usually very good.

    # 4. Construct vectors
    vectors = []
    # Iterate through best_set
    for s_idx in best_set:
        support = supports[s_idx]
        # Generate all 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)

    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
