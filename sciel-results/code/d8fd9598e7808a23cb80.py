import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 148  # safety margin before 150s limit
    
    # Generate all D_11 vectors with squared norm 4
    candidates = []
    
    # Type 1: One ±2, rest 0 → 2*11 = 22 vectors
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four ±1's, rest 0 → C(11,4) * 2^4 = 330 * 16 = 5280
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict: two vectors are incompatible if M[i,j] > 2
    conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Greedy initialization with multiple orderings
    rng = np.random.default_rng(42)
    
    compat = (M <= 2).astype(np.uint8)
    compat_count = compat.sum(axis=1)
    
    best_set = []
    best_size = 0
    
    # Multiple greedy passes with different orderings
    orderings = [np.argsort(-compat_count), np.arange(N)]
    for _ in range(50):
        orderings.append(rng.permutation(N))
    
    for order in orderings:
        if time.time() > deadline - 60:
            break
            
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            dots = M[idx, selected_mask]
            if np.all(dots <= 2):
                selected.append(idx)
                selected_mask[idx] = True
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # Simulated annealing with swap moves
    if time.time() < deadline - 10 and best_set:
        in_set = np.zeros(N, dtype=bool)
        in_set[best_set] = True
        current_size = len(best_set)
        
        # Precompute for each vertex, the number of conflicts with current set
        # conflict_count[i] = number of selected vertices that conflict with i
        conflict_counts = conflict[:, in_set].sum(axis=1)
        
        # Vertices that can be added: conflict_counts[i] == 0 and not in_set
        # Vertices that can be swapped: remove j from set, then add i if i is compatible with new set
        
        best_sa_size = current_size
        best_sa_set = best_set[:]
        
        T = 50.0
        T_min = 0.1
        cooling = 0.9995
        
        sa_steps = 0
        while time.time() < deadline - 5 and T > T_min:
            # Pick a random move: add, remove, or swap
            move_type = rng.random()
            
            if move_type < 0.3:
                # Try to add a random unselected vertex
                unselected = np.where(~in_set)[0]
                if len(unselected) == 0:
                    continue
                i = unselected[rng.integers(len(unselected))]
                
                if conflict_counts[i] == 0:
                    # Can add
                    in_set[i] = True
                    current_size += 1
                    # Update conflict counts for all unselected
                    conflict_counts[~in_set] += conflict[i, ~in_set]
                    
                    if current_size > best_sa_size:
                        best_sa_size = current_size
                        best_sa_set = np.where(in_set)[0].tolist()
                else:
                    # Cannot add, no change
                    pass
                    
            elif move_type < 0.6:
                # Try to remove a random selected vertex
                selected_arr = np.where(in_set)[0]
                if len(selected_arr) == 0:
                    continue
                j = selected_arr[rng.integers(len(selected_arr))]
                
                # Remove j
                in_set[j] = False
                current_size -= 1
                conflict_counts[~in_set] -= conflict[j, ~in_set]
                
                # With some probability, accept the removal (to explore)
                if rng.random() > 0.5:
                    # Keep removed, try to add something else
                    pass
                else:
                    # Re-add j
                    in_set[j] = True
                    current_size += 1
                    conflict_counts[~in_set] += conflict[j, ~in_set]
                    
            else:
                # Swap: remove one selected, add one unselected
                selected_arr = np.where(in_set)[0]
                unselected = np.where(~in_set)[0]
                if len(selected_arr) == 0 or len(unselected) == 0:
                    continue
                
                j = selected_arr[rng.integers(len(selected_arr))]
                i = unselected[rng.integers(len(unselected))]
                
                # Check if i is compatible with all selected except j
                # i conflicts with j? If yes, removing j helps
                # i must not conflict with any other selected vertex
                selected_except_j = np.where(in_set & (np.arange(N) != j))[0]
                
                if len(selected_except_j) > 0:
                    dots = M[i, selected_except_j]
                    if np.any(dots > 2):
                        # i conflicts with some other selected, skip
                        continue
                
                # Perform swap: remove j, add i
                in_set[j] = False
                in_set[i] = True
                # current_size unchanged
                
                # Update conflict counts
                # Remove j's contribution, add i's contribution
                conflict_counts[~in_set] += conflict[i, ~in_set] - conflict[j, ~in_set]
                
                # Accept with some probability based on temperature
                # Since size doesn't change, accept based on "quality"
                # Use a simple acceptance: always accept swaps that maintain validity
                # (which they do by construction)
                
            sa_steps += 1
            T *= cooling
            
            # Periodically try to add all possible vertices
            if sa_steps % 1000 == 0 and time.time() < deadline - 5:
                unselected = np.where(~in_set)[0]
                for i in unselected:
                    if conflict_counts[i] == 0:
                        in_set[i] = True
                        current_size += 1
                        conflict_counts[~in_set] += conflict[i, ~in_set]
                
                if current_size > best_sa_size:
                    best_sa_size = current_size
                    best_sa_set = np.where(in_set)[0].tolist()
    
    if best_sa_size > best_size:
        best_size = best_sa_size
        best_set = best_sa_set
    
    # Final greedy expansion
    if time.time() < deadline - 3 and best_set:
        in_set = np.zeros(N, dtype=bool)
        in_set[best_set] = True
        for idx in range(N):
            if in_set[idx]:
                continue
            dots = M[idx, in_set]
            if np.all(dots <= 2):
                best_set.append(idx)
                in_set[idx] = True
                best_size = len(best_set)
    
    if not best_set:
        # Fallback: just use the 22 weight-2 vectors
        result = candidates[:22].astype(np.int64)
        return result
        
    result = candidates[best_set].astype(np.int64)
    
    # Verify validity
    n = len(result)
    if n == 0:
        return candidates[:22].astype(np.int64)
    
    max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
    # Vectorized check
    diffs = result[:, np.newaxis, :] - result[np.newaxis, :, :]
    diff_norms_sq = np.sum(diffs**2, axis=2)
    if np.min(diff_norms_sq[np.triu_indices(n, k=1)]) < max_norm_sq:
        # Invalid, fallback
        return candidates[:22].astype(np.int64)
    
    return result
