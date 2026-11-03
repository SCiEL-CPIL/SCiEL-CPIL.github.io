import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)  # 330
    
    # Precompute conflict matrix using bitmasks for speed
    # Each support is a 4-element tuple; convert to bitmask
    support_masks = np.array([sum(1 << i for i in s) for s in supports], dtype=np.int64)
    
    # Conflict: two supports share >= 3 indices
    # For 4-element sets, sharing >= 3 means intersection size is 3 or 4
    # intersection size = popcount(mask_i & mask_j)
    # Since we're using 11 bits, we can compute pairwise
    
    # Vectorized: compute all pairwise intersections
    # This is 330x330 = 108900 entries, manageable
    # For each pair, compute popcount of AND
    
    # Use numpy for vectorized computation
    # Create upper triangle indices for efficiency
    iu = np.triu_indices(n_supports, k=1)
    pairs = (support_masks[iu[0]] & support_masks[iu[1]])
    
    # Count bits in each pair
    # Use lookup table for 11-bit numbers (max value 2047)
    popcount_lut = np.array([bin(i).count('1') for i in range(2048)], dtype=np.int8)
    intersections = popcount_lut[pairs]
    
    # Conflict if intersection >= 3
    conflict_pairs = np.where(intersections >= 3)[0]
    
    # Build conflict matrix
    conflict = np.zeros((n_supports, n_supports), dtype=bool)
    conflict[iu[0][conflict_pairs], iu[1][conflict_pairs]] = True
    conflict[iu[1][conflict_pairs], iu[0][conflict_pairs]] = True
    
    # Precompute degree
    degree = np.sum(conflict, axis=1)
    
    # Precompute adjacency lists for fast neighbor lookup
    neighbors = [np.where(conflict[i])[0] for i in range(n_supports)]
    
    best_size = 0
    best_selection = np.zeros(n_supports, dtype=bool)
    
    rng = np.random.default_rng(42)
    
    # Simulated annealing with multiple restarts
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize: greedy MIS
        in_set = np.zeros(n_supports, dtype=bool)
        remaining = np.ones(n_supports, dtype=bool)
        
        # Greedy initialization with random tie-breaking
        while np.any(remaining):
            if time.time() > deadline:
                break
            rem_idx = np.where(remaining)[0]
            if len(rem_idx) == 0:
                break
            
            # Pick node with minimum degree among remaining, with random noise
            degs = degree[rem_idx]
            noise = rng.exponential(0.5, size=len(rem_idx))
            scores = degs + noise
            min_idx = np.argmin(scores)
            chosen = rem_idx[min_idx]
            
            in_set[chosen] = True
            remaining[chosen] = False
            remaining[neighbors[chosen]] = False
        
        current_size = np.sum(in_set)
        
        if current_size > best_size:
            best_size = current_size
            best_selection = in_set.copy()
        
        # Simulated annealing phase
        if current_size == 0:
            continue
        
        # Temperature schedule
        T_start = 2.0
        T_end = 0.01
        n_sa_steps = 5000
        alpha = (T_end / T_start) ** (1.0 / n_sa_steps)
        
        T = T_start
        
        # Precompute for fast energy calculation
        # Energy = -size (we want to maximize size, so minimize -size)
        # For a swap: remove node i, add node j
        # Valid if j is not in set, and j doesn't conflict with any other node in set
        # Delta energy = -1 + 1 = 0 if valid swap (size unchanged)
        # But we can also do: remove i, don't add j (size -1) or add j without removing (size +1 if valid)
        
        # Better: try single additions and single removals and swaps
        
        # Get current set members
        set_members = np.where(in_set)[0]
        
        for step in range(n_sa_steps):
            if time.time() > deadline:
                break
            
            T *= alpha
            
            # Choose move type: add, remove, or swap
            move_type = rng.integers(0, 3)
            
            if move_type == 0:  # Try to add a node
                # Find nodes not in set
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                
                # Pick random candidate
                cand = not_in_set[rng.integers(0, len(not_in_set))]
                
                # Check if cand conflicts with any node in set
                # cand can be added if it doesn't conflict with any set member
                conflicts_with_set = np.sum(conflict[cand] & in_set)
                
                if conflicts_with_set == 0:
                    # Valid addition, always accept
                    in_set[cand] = True
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_selection = in_set.copy()
                else:
                    # Invalid: would create conflict
                    # Energy increase = conflicts_with_set (number of conflicts created)
                    # Actually, for independent set, we can't add if any conflict exists
                    # So this move is always rejected (infinite energy)
                    pass
                    
            elif move_type == 1:  # Try to remove a node
                if len(set_members) == 0:
                    continue
                # Pick random node from set
                idx = rng.integers(0, len(set_members))
                node = set_members[idx]
                
                # Removing always reduces size by 1
                # Accept with probability exp(-1/T) (energy increase of 1)
                if rng.random() < np.exp(-1.0 / T):
                    in_set[node] = False
                    current_size -= 1
                    set_members = np.where(in_set)[0]
                    
            else:  # Try swap: remove one, add one
                if len(set_members) == 0:
                    continue
                
                # Pick node to remove
                idx = rng.integers(0, len(set_members))
                remove_node = set_members[idx]
                
                # Find candidates to add: not in set, and don't conflict with set minus remove_node
                # A candidate j is valid if:
                # 1. j not in set
                # 2. j doesn't conflict with any set member except possibly remove_node
                
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                
                # For each candidate, count conflicts with (set - {remove_node})
                # Create a temporary in_set without remove_node
                in_set_temp = in_set.copy()
                in_set_temp[remove_node] = False
                
                # Count conflicts for all candidates at once
                # conflicts = sum(conflict[cand] & in_set_temp) for each cand
                # Vectorized: use matrix multiplication
                cand_conflicts = np.sum(conflict[not_in_set] & in_set_temp[None, :], axis=1)
                
                # Valid candidates: those with 0 conflicts
                valid_cands = not_in_set[cand_conflicts == 0]
                
                if len(valid_cands) > 0:
                    # Pick random valid candidate
                    add_node = valid_cands[rng.integers(0, len(valid_cands))]
                    
                    # Swap: remove remove_node, add add_node
                    # Size unchanged, so always accept (neutral move)
                    in_set[remove_node] = False
                    in_set[add_node] = True
                    set_members = np.where(in_set)[0]
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_selection = in_set.copy()
                else:
                    # No valid swap partner; try to just remove
                    if rng.random() < np.exp(-1.0 / T):
                        in_set[remove_node] = False
                        current_size -= 1
                        set_members = np.where(in_set)[0]
        
        if current_size > best_size:
            best_size = current_size
            best_selection = in_set.copy()
    
    # Construct vectors from best selection
    vectors = []
    for s_idx in range(n_supports):
        if best_selection[s_idx]:
            support = supports[s_idx]
            # Generate all 16 sign variants
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int64)
                for k, idx in enumerate(support):
                    v[idx] = signs[k]
                vectors.append(v)
    
    if len(vectors) == 0:
        # Fallback: return a single vector
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    return np.array(vectors, dtype=np.int64)
