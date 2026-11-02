"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to stay within 150s wall-clock
    
    # Generate candidate vectors: all vectors with entries in {-1, 0, 1}
    # that have at most 3 non-zero entries (to keep the search space manageable)
    # This includes the seed's 2-nonzero vectors plus 1-nonzero and 3-nonzero vectors
    
    candidates = []
    
    # 1-nonzero vectors: 11 * 2 = 22
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(tuple(v))
    
    # 2-nonzero vectors: C(11,2) * 4 = 55 * 4 = 220
    from itertools import combinations
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(tuple(v))
    
    # 3-nonzero vectors: C(11,3) * 8 = 165 * 8 = 1320
    for i, j, k in combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(tuple(v))
    
    # Convert to numpy array
    cand_arr = np.array(candidates, dtype=np.int64)
    n_cand = len(cand_arr)
    
    # Precompute squared norms
    norms_sq = np.sum(cand_arr ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Greedy set packing with simulated annealing swaps
    # A set is valid if for all pairs u, v in the set:
    #   ||u - v||^2 >= max_{w in set} ||w||^2
    # Since we normalize by max norm, the condition is:
    #   ||u - v|| >= max_norm  (for all pairs)
    # Equivalently: ||u - v||^2 >= max_norm^2
    
    # Start with empty set
    selected = []  # list of indices into cand_arr
    selected_set = set()
    
    # Greedy initial construction
    # Sort candidates by norm (prefer lower norm to make packing easier)
    order = np.argsort(norms_sq)
    
    for idx in order:
        if time.time() > deadline - 5:
            break
        v = cand_arr[idx]
        # Check if adding v maintains validity
        valid = True
        for sel_idx in selected_set:
            diff = v - cand_arr[sel_idx]
            if np.dot(diff, diff) < max_norm_sq:
                valid = False
                break
        if valid:
            selected_set.add(idx)
            selected.append(idx)
    
    best_size = len(selected)
    best_selected = list(selected)
    
    # Simulated annealing to improve
    # Temperature schedule
    t_start = 1.0
    t_end = 0.01
    n_iterations = 50000
    
    # Precompute pairwise distances for candidates in selected set
    # For efficiency, we'll work with the current set
    
    current_selected = list(best_selected)
    current_set = set(current_selected)
    current_size = len(current_selected)
    
    # Precompute norms for quick access
    cand_norms_sq = norms_sq  # already computed
    
    # For SA, we need to evaluate swaps: remove one vector, add another
    # To speed up, precompute which candidates are "compatible" with each selected vector
    
    # Let's do a simpler SA: try to add a random candidate, if it conflicts, 
    # try removing one of the conflicting vectors
    
    t = t_start
    cooling = (t_end / t_start) ** (1.0 / max(n_iterations, 1))
    
    for it in range(n_iterations):
        if time.time() > deadline - 5:
            break
        
        t *= cooling
        
        # Try to add a random candidate not in the set
        if len(current_set) < n_cand:
            # Pick a random candidate index
            cand_idx = random.randint(0, n_cand - 1)
            if cand_idx not in current_set:
                v = cand_arr[cand_idx]
                # Check conflicts
                conflicts = []
                for sel_idx in current_set:
                    diff = v - cand_arr[sel_idx]
                    if np.dot(diff, diff) < max_norm_sq:
                        conflicts.append(sel_idx)
                
                if not conflicts:
                    # Can add without removing
                    current_set.add(cand_idx)
                    current_selected.append(cand_idx)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_selected = list(current_selected)
                elif conflicts:
                    # Try to remove one conflicting vector and add the new one
                    # Only beneficial if we can add (size stays same or increases)
                    # For now, just try swapping: remove one conflict, add new
                    if len(conflicts) > 0:
                        rem_idx = random.choice(conflicts)
                        # Check if removing rem_idx and adding cand_idx is valid
                        # After removal, need to check cand_idx against all remaining
                        new_conflicts = []
                        for sel_idx in current_set:
                            if sel_idx == rem_idx:
                                continue
                            diff = v - cand_arr[sel_idx]
                            if np.dot(diff, diff) < max_norm_sq:
                                new_conflicts.append(sel_idx)
                        
                        if not new_conflicts:
                            # Valid swap
                            current_set.discard(rem_idx)
                            current_set.add(cand_idx)
                            # Update current_selected
                            if rem_idx in current_selected:
                                current_selected.remove(rem_idx)
                            current_selected.append(cand_idx)
                            # Size unchanged, but might enable future additions
        
        # Occasionally try removing a vector to see if it enables more additions
        if current_size > 0 and random.random() < 0.1:
            rem_idx = random.choice(list(current_set))
            current_set.discard(rem_idx)
            if rem_idx in current_selected:
                current_selected.remove(rem_idx)
            current_size -= 1
            
            # Try to add back multiple vectors
            for cand_idx in order:
                if cand_idx in current_set:
                    continue
                v = cand_arr[cand_idx]
                valid = True
                for sel_idx in current_set:
                    diff = v - cand_arr[sel_idx]
                    if np.dot(diff, diff) < max_norm_sq:
                        valid = False
                        break
                if valid:
                    current_set.add(cand_idx)
                    current_selected.append(cand_idx)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_selected = list(current_selected)
    
    # Return the best set found
    if best_selected:
        return cand_arr[best_selected]
    else:
        # Fallback to seed
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
