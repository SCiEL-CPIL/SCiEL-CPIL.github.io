import numpy as np
import time
import itertools
import random

D = 11

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # 1. Generate Candidate Pool
    # We use vectors with squared norms 3, 4, 5.
    # Condition: min pairwise dist >= max norm.
    # If max norm is sqrt(5), min dist >= sqrt(5).
    # If we stick to sqrt(4), min dist >= 2.
    # Mixed norms allow more flexibility.
    
    candidates = []
    
    # Norm^2 = 3: 3 non-zeros of +/-1
    # C(11,3) * 2^3 = 165 * 8 = 1320
    for idxs in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for i, s in zip(idxs, signs):
                v[i] = s
            candidates.append(v)
            
    # Norm^2 = 4: 
    # Case A: 2 non-zeros of +/-2 (C(11,2)*4 = 55*4=220)
    # Case B: 4 non-zeros of +/-1 (C(11,4)*16 = 330*16=5280)
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([2, -2], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for i, s in zip(idxs, signs):
                v[i] = s
            candidates.append(v)
            
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for i, s in zip(idxs, signs):
                v[i] = s
            candidates.append(v)
            
    # Norm^2 = 5:
    # Case A: 1 non-zero of +/-1, 2 non-zeros of +/-2? No, 1+4=5.
    # Indices: 1 for '1', 2 for '2'. C(11,1)*C(10,2)*2^3 = 11*45*8 = 3960
    # Case B: 5 non-zeros of +/-1. C(11,5)*32 = 462*32 = 14784
    # Total for 5 is large. Let's include only Case A to keep pool manageable?
    # Actually, weight 5 vectors have dist >= sqrt(5) if max norm is sqrt(5).
    # Let's include Case B as well if time permits, but Case A is good.
    # Let's stick to Norm^2 3 and 4 first, maybe add 5 if needed.
    # The inspiration says weight-4 (norm 2) is good. 
    # Let's add Norm^2=5 Case B (5 ones) because they are "sparse" in a way? No, they are denser.
    # Let's just use 3 and 4 for now, it's ~7000 candidates.
    
    # Actually, let's add Norm^2=5 Case A (1x1, 2x2)
    for i1 in range(D):
        for i2, i3 in itertools.combinations([x for x in range(D) if x != i1], 2):
            for s1 in [1, -1]:
                for s2, s3 in itertools.product([2, -2], repeat=2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i1] = s1
                    v[i2] = s2
                    v[i3] = s3
                    candidates.append(v)
    
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Precompute Gram Matrix G where G[i,j] = v_i . v_j
    # dist^2(i,j) = ||v_i||^2 + ||v_j||^2 - 2 G[i,j]
    G = candidates @ candidates.T
    norms_sq = np.sum(candidates**2, axis=1)
    
    # 2. Simulated Annealing
    # State: A set of indices S.
    # Objective: Maximize |S| subject to constraint.
    # Constraint: For all i,j in S, dist(i,j)^2 >= max_norm_sq(S)
    
    # Initialize with a greedy solution
    # Greedy: Sort by norm_sq ascending? Or random?
    # Let's try to build a large set greedily first to get a good start.
    
    def greedy_init():
        # Random order
        order = np.random.permutation(n_cand)
        selected = []
        max_norm_sq = 0
        for idx in order:
            if not selected:
                selected.append(idx)
                max_norm_sq = norms_sq[idx]
                continue
            
            # Check if adding idx violates constraint with current selected
            # New max_norm_sq = max(max_norm_sq, norms_sq[idx])
            new_max = max(max_norm_sq, norms_sq[idx])
            
            # Check dist to all selected
            # dist^2 = norms_sq[idx] + norms_sq[sel] - 2*G[idx, sel]
            # We need dist^2 >= new_max for all sel
            sel_arr = np.array(selected)
            d2 = norms_sq[idx] + norms_sq[sel_arr] - 2 * G[idx, sel_arr]
            
            if np.all(d2 >= new_max):
                selected.append(idx)
                max_norm_sq = new_max
        
        return selected, max_norm_sq
    
    # Run a few greedy inits to find a good starting point
    best_init_size = 0
    best_init_set = []
    best_init_max_norm = 0
    
    for _ in range(20):
        if time.time() > deadline - 60: break
        s, mn = greedy_init()
        if len(s) > best_init_size:
            best_init_size = len(s)
            best_init_set = s
            best_init_max_norm = mn
    
    if not best_init_set:
        # Fallback
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    current_set = set(best_init_set)
    current_max_norm_sq = best_init_max_norm
    current_size = len(current_set)
    
    best_set = list(current_set)
    best_size = current_size
    
    # Annealing Parameters
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9995
    
    # We need to efficiently check constraint.
    # When we add/remove a vector, we need to update max_norm_sq and check distances.
    # Since max_norm_sq is global to the set, it's tricky.
    # Simplification: 
    # If we add a vector with norm > current max, new max is that norm. 
    # If we remove the vector that defined the max, we need to recalculate max.
    
    # To make it faster, we can restrict moves to:
    # 1. Add a vector (if it doesn't increase max_norm, or if it does, check all).
    # 2. Remove a vector.
    # 3. Swap a vector.
    
    # Let's stick to Add/Remove for simplicity, maybe Swap later.
    
    while time.time() < deadline - 10:
        if T < T_min:
            break
            
        # Pick a move
        if random.random() < 0.5:
            # Try Add
            # Pick a candidate not in set
            # To speed up, we can pre-filter candidates that are "compatible" with current max?
            # Just pick random.
            idx = np.random.randint(n_cand)
            if idx in current_set:
                continue
                
            new_max = max(current_max_norm_sq, norms_sq[idx])
            
            # Check compatibility with all in current_set
            # We need dist^2 >= new_max for all j in current_set
            # d2_j = norms_sq[idx] + norms_sq[j] - 2 G[idx, j]
            
            # Vectorized check against current_set
            if current_set:
                sel_list = list(current_set)
                sel_arr = np.array(sel_list)
                d2 = norms_sq[idx] + norms_sq[sel_arr] - 2 * G[idx, sel_arr]
                if not np.all(d2 >= new_max):
                    continue # Reject if constraint violated
            
            # Accept Add
            current_set.add(idx)
            current_max_norm_sq = new_max
            current_size += 1
            
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
                
        else:
            # Try Remove
            if not current_set:
                continue
            idx = random.choice(list(current_set))
            
            # Remove
            current_set.remove(idx)
            current_size -= 1
            
            # Recalculate max_norm_sq
            # This is O(N). To avoid this, we can keep a count of norms or just recalc if necessary.
            # For simplicity, recalc.
            if current_set:
                current_max_norm_sq = max(norms_sq[i] for i in current_set)
            else:
                current_max_norm_sq = 0
            
            # Accept Remove (always valid, but might be bad)
            # We don't update best here unless size is larger (it's not)
            
        # Cooling
        T *= cooling_rate
        
    # Finalize
    if not best_set:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return candidates[np.array(best_set)]

# Run construct to test? No, just define it.
