import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type B: Vectors with two non-zero entries +/- 1
    vectors_B = []
    for i in range(D):
        for j in range(i+1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    vectors_B.append(v)
    vectors_B = np.array(vectors_B, dtype=np.int64)

    # Type D: Vectors with four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Build conflict graph for supports
    # Conflict if intersection size >= 3
    support_masks = np.zeros(N_S, dtype=np.int32)
    for s_idx, s in enumerate(supports):
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks[s_idx] = mask

    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(1, 2048):
        popcount[i] = popcount[i >> 1] + (i & 1)

    conflict = np.zeros((N_S, N_S), dtype=bool)
    # Vectorized computation of conflict
    # inter = support_masks[:, None] & support_masks[None, :]
    # This creates a 330x330 array of ints, which is fine.
    inter = support_masks[:, None] & support_masks[None, :]
    pc = popcount[inter]
    conflict = (pc >= 3)
    # Exclude self
    np.fill_diagonal(conflict, False)

    # Find Max Independent Set using Simulated Annealing
    rng = np.random.RandomState(42)
    
    # Greedy Init
    x = np.zeros(N_S, dtype=bool)
    order = np.argsort(conflict.sum(axis=1))
    for v in order:
        if not x[v] and not conflict[v].any():
            x[v] = True
    
    best_x = x.copy()
    best_size = int(np.sum(x))
    
    T0 = 1.0
    T_end = 0.01
    num_iterations = 200000
    
    # Precompute neighbors for faster checking?
    # conflict[v] is a boolean array.
    
    for it in range(num_iterations):
        if time.time() > deadline - 10:
            break
            
        T = T0 * (T_end / T0) ** (it / num_iterations)
        
        # Pick random node
        v = rng.randint(N_S)
        
        if x[v]:
            # Remove
            x[v] = False
            new_size = int(np.sum(x))
            # Always accept removal? No, might want to keep it.
            # But removal is always valid.
            # We accept with prob 1 if size increases (impossible here), 
            # or if we want to explore.
            # Let's just accept removal to allow exploration.
            # Actually, standard SA: 
            # Delta = -1.
            # Accept if rand < exp(1/T) ? No, exp(-Delta/T) = exp(1/T) > 1. Always accept?
            # Usually we penalize size decrease.
            # If we remove, size decreases by 1.
            # We should only accept removal if it allows adding more later?
            # Simplest: Accept removal with prob exp(1/T) (always > 1, so always accept).
            # This makes it a random walk that tends to shrink?
            # Better: Only remove if it helps?
            # Let's use a different move: Swap.
            
            # Let's stick to Add/Remove but be careful.
            # If we remove, we lose 1.
            # We should only accept if we expect to gain more.
            # For now, let's just do Add moves and Swap moves.
            pass # Skip simple remove for now, use swap logic below
            
        else:
            # Try to add
            if not conflict[v].any():
                x[v] = True
                new_size = int(np.sum(x))
                if new_size > best_size:
                    best_size = new_size
                    best_x = x.copy()
            else:
                # Try swap: remove one neighbor, add v
                neighbors = np.where(conflict[v] & x)[0]
                if len(neighbors) > 0:
                    # Pick random neighbor to remove
                    rem_idx = neighbors[rng.randint(len(neighbors))]
                    # Check if removing rem_idx and adding v is valid
                    # v is compatible with all non-neighbors.
                    # rem_idx is a neighbor of v.
                    # Are there other conflicts?
                    # v conflicts with neighbors. rem_idx is one of them.
                    # If we remove rem_idx, v is compatible with rest of x?
                    # Yes, because v only conflicts with neighbors in x.
                    # And we removed the only one we're considering?
                    # Wait, v might conflict with MULTIPLE neighbors in x.
                    # If so, we can't add v unless we remove ALL of them.
                    # This is complex.
                    
                    # Simpler: Just do Add/Remove.
                    # To avoid shrinking, only accept removal with prob exp(1/T) is bad.
                    # Let's try:
                    # If add is blocked, try to remove a random selected node.
                    # This allows escaping local optima.
                    selected = np.where(x)[0]
                    if len(selected) > 0:
                        rem_idx = selected[rng.randint(len(selected))]
                        x[rem_idx] = False
                        # Accept removal with probability based on T?
                        # Let's just accept it to allow exploration.
                        # This is a "kick" move.
                        
    # Generate vectors for selected supports
    selected_supports = np.where(best_x)[0]
    
    vectors_D_selected = []
    for s_idx in selected_supports:
        s = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            vectors_D_selected.append(v)
            
    vectors_D_selected = np.array(vectors_D_selected, dtype=np.int64)
    
    final_vectors = np.vstack([vectors_A, vectors_B, vectors_D_selected])
    return final_vectors
