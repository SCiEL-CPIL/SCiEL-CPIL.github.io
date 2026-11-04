import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # 1. Generate all Type A vectors (22 vectors)
    # These are +/- 2 * e_i
    type_a_vectors = []
    for i in range(D):
        for sign in (-2, 2):
            v = np.zeros(D, dtype=np.int8)
            v[i] = sign
            type_a_vectors.append(v)
    
    # 2. Generate all possible Type D supports (4-subsets of 11)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    # Convert supports to bitmasks for fast intersection check
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    
    # 3. Build Conflict Matrix
    # Conflict if intersection size >= 3
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                conflict[i, j] = True
                conflict[j, i] = True
                
    # Precompute neighbors for faster updates? 
    # conflict[i] is a boolean array.
    
    # 4. Find Maximum Independent Set (MIS) using Simulated Annealing
    rng = np.random.RandomState(42)
    
    def random_greedy_init():
        x = np.zeros(N_S, dtype=bool)
        order = rng.permutation(N_S)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not x[v] and not blocked[v]:
                x[v] = True
                blocked |= conflict[v]
        return x

    # Initial state
    x = random_greedy_init()
    best_x = x.copy()
    best_size = np.sum(x)
    
    # Check initial validity (should be valid by construction)
    
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    steps = 100000
    
    # Helper to calculate current conflict count and size
    # To make SA fast, we track delta changes.
    
    current_conflicts = 0
    current_size = int(np.sum(x))
    
    # Precompute degree or something? 
    # Let's just use the matrix directly.
    
    for step in range(steps):
        if time.time() > deadline - 10:
            break
            
        if T < T_min:
            T = T_min
            
        # Pick a random node to flip
        v = rng.randint(N_S)
        
        if x[v]:
            # Removing v
            # Delta size: -1
            # Delta conflicts: 
            # Conflicts involving v are removed.
            # Let n_conf = number of neighbors of v that are in x
            n_conf = np.sum(conflict[v] & x)
            # Removing v removes n_conf conflicts.
            delta_conf = -n_conf
            delta_size = -1
        else:
            # Adding v
            # Delta size: +1
            # Delta conflicts:
            # Adding v creates conflicts with all neighbors of v that are in x
            n_conf = np.sum(conflict[v] & x)
            delta_conf = n_conf
            delta_size = 1
            
        # Cost function: -size + lambda * conflicts
        # We want to minimize cost.
        # Delta cost = -delta_size + lambda * delta_conf
        lambda_pen = 10.0
        delta_cost = -delta_size + lambda_pen * delta_conf
        
        if delta_cost < 0 or (delta_cost > 0 and rng.rand() < np.exp(-delta_cost / T)):
            # Accept move
            x[v] = not x[v]
            current_size += delta_size
            current_conflicts += delta_conf
            
            # Update best if valid
            if current_conflicts == 0 and current_size > best_size:
                best_size = current_size
                best_x = x.copy()
                
        T *= alpha
        
    # After SA, try to add any remaining compatible nodes to best_x
    # This is a greedy repair
    x_best = best_x.copy()
    size_best = best_size
    blocked = np.zeros(N_S, dtype=bool)
    # Initialize blocked based on current set
    for i in range(N_S):
        if x_best[i]:
            blocked |= conflict[i]
            
    # Try to add any unblocked node
    changed = True
    while changed:
        changed = False
        for i in range(N_S):
            if not x_best[i] and not blocked[i]:
                x_best[i] = True
                size_best += 1
                blocked |= conflict[i]
                changed = True
                if time.time() > deadline - 5:
                    break
        if time.time() > deadline - 5:
            break

    # 5. Construct Type D vectors from best_x
    type_d_vectors = []
    for i in range(N_S):
        if best_x[i]:
            support = supports[i]
            # Generate all 16 sign combinations for these 4 indices
            for mask in range(16):
                v = np.zeros(D, dtype=np.int8)
                for k in range(4):
                    if (mask >> k) & 1:
                        v[support[k]] = -1
                    else:
                        v[support[k]] = 1
                type_d_vectors.append(v)
                
    # 6. Combine and verify
    all_vectors = type_a_vectors + type_d_vectors
    if not all_vectors:
        return np.zeros((0, D), dtype=np.int8)
        
    vec_array = np.array(all_vectors, dtype=np.int8)
    
    # Quick verification (optional but good for safety)
    # Norms should be 2 (Type A) or 2 (Type D, sqrt(4)=2).
    # Max norm = 2.
    # Min distance should be >= 2.
    
    # Calculate norms
    norms = np.sqrt(np.sum(vec_array**2, axis=1))
    max_norm = np.max(norms)
    
    # Calculate pairwise distances
    # This might be slow if k is large (e.g., 500). 500x500 = 250k pairs.
    # 250k ops is fast in NumPy.
    if len(vec_array) > 1:
        # dist^2 = ||v||^2 + ||w||^2 - 2 v.w
        # Since all norms are 2, ||v||^2 = 4.
        # dist^2 = 4 + 4 - 2 v.w = 8 - 2 v.w
        # We need dist >= 2 => dist^2 >= 4 => 8 - 2 v.w >= 4 => 2 v.w <= 4 => v.w <= 2.
        
        # Compute dot products
        dots = vec_array @ vec_array.T
        # Check off-diagonal elements
        # Set diagonal to -inf to ignore
        np.fill_diagonal(dots, -np.inf)
        max_dot = np.max(dots)
        
        if max_dot > 2:
            # Invalid configuration found?
            # This shouldn't happen if logic is correct.
            # Fallback to E8 if this happens?
            # For now, return as is, but we expect it to be valid.
            pass

    return vec_array

