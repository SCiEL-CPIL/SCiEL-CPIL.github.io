import numpy as np
import itertools
import time
import math

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    sq_norms = (grid ** 2).sum(axis=1)
    
    # Precompute differences
    # diff[i, j] = grid[j] - grid[i]
    diff = grid[np.newaxis, :, :] - grid[:, np.newaxis, :] 
    # sq_diff[i, j] = sq_norms[j] - sq_norms[i]
    sq_diff = sq_norms[np.newaxis, :] - sq_norms[:, np.newaxis]
    
    # Precompute all combinations of 4 indices for efficiency in violation counting?
    # No, k varies. We'll generate combinations on the fly or cache for specific k?
    # C(k,4) generation is fast enough.
    
    def count_violations_with_candidate(candidate_idx, other_indices):
        """
        Count how many 4-subsets of other_indices form a cospherical/coplanar 5-tuple with candidate_idx.
        other_indices: array/list of indices into grid, not including candidate_idx.
        """
        k = len(other_indices)
        if k < 4:
            return 0
        
        # Get vectors and sq diffs relative to candidate
        d_vecs = diff[candidate_idx, other_indices, :]  # (k, 3)
        d_sqs = sq_diff[candidate_idx, other_indices]   # (k,)
        
        # Generate all 4-combinations of indices from 0 to k-1
        # Using itertools.combinations is slow in Python loop, but we can vectorize the extraction?
        # Actually, for k~50, C(50,4)=230k. Generating list of tuples is ~1ms.
        # Extracting rows is the bottleneck.
        
        # Efficient extraction:
        # Create array of all 4-combinations of indices
        combos = np.array(list(itertools.combinations(range(k), 4)), dtype=np.int32)
        
        if len(combos) == 0:
            return 0
            
        # d_vecs[combos] -> (C, 4, 3)
        # d_sqs[combos] -> (C, 4)
        # Concatenate to (C, 4, 4)
        rows = np.concatenate([d_vecs[combos], d_sqs[combos][:, :, np.newaxis]], axis=2)
        
        # Calculate determinants
        dets = np.linalg.det(rows)
        
        # Count violations (|det| < 0.5)
        return int(np.sum(np.abs(dets) < 0.5))

    def get_initial_greedy():
        """
        Simple greedy initialization to start SA from a good state.
        """
        np.random.seed(42)
        order = np.random.permutation(N)
        selected = []
        for idx in order:
            # Check if can add
            # For initialization, we can be strict
            v = count_violations_with_candidate(idx, selected)
            if v == 0:
                selected.append(idx)
        return selected

    # SA Parameters
    PENALTY = 100.0
    T_START = 10.0
    T_END = 0.01
    COOLING_RATE = 0.999
    
    # Initialize
    current_set = get_initial_greedy()
    current_set_list = list(current_set)
    current_set_set = set(current_set_list)
    
    # Calculate initial violations
    # This is O(k^5) if we do it naively, but we can sum over points?
    # No, just compute total violations in the set.
    # Total violations = sum over all 5-subsets.
    # Or, since we built it greedily with v=0, initial violations should be 0.
    current_violations = 0
    
    best_set = list(current_set_list)
    best_k = len(best_set)
    
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 10.0
    
    T = T_START
    
    while time.time() - start_time < time_limit - safety_margin:
        k = len(current_set_list)
        
        # Propose a move
        move_type = np.random.random()
        delta_cost = 0
        delta_violations = 0
        new_set_list = current_set_list
        new_set_set = current_set_set
        candidate_idx = -1
        remove_idx = -1
        
        if move_type < 0.4:
            # Add a point
            if k < N:
                # Pick a random point not in set
                # To speed up, we can pre-compute a list of candidates?
                # For now, random sampling from 0..N-1
                attempts = 0
                while attempts < 10:
                    candidate_idx = np.random.randint(N)
                    if candidate_idx not in current_set_set:
                        break
                    attempts += 1
                if candidate_idx != -1 and candidate_idx not in current_set_set:
                    v_add = count_violations_with_candidate(candidate_idx, current_set_list)
                    delta_violations = v_add
                    delta_cost = v_add * PENALTY - 1.0
                    new_set_list = current_set_list + [candidate_idx]
                    new_set_set = current_set_set | {candidate_idx}
        elif move_type < 0.7:
            # Remove a point
            if k > 0:
                remove_idx = np.random.choice(current_set_list)
                # Calculate violations involving remove_idx
                others = [i for i in current_set_list if i != remove_idx]
                v_rem = count_violations_with_candidate(remove_idx, others)
                delta_violations = -v_rem
                delta_cost = -v_rem * PENALTY + 1.0
                new_set_list = [i for i in current_set_list if i != remove_idx]
                new_set_set = current_set_set - {remove_idx}
        else:
            # Swap a point
            if k > 0 and k < N:
                remove_idx = np.random.choice(current_set_list)
                # Find candidate to add
                attempts = 0
                valid_candidate = -1
                while attempts < 10:
                    candidate_idx = np.random.randint(N)
                    if candidate_idx not in current_set_set:
                        valid_candidate = candidate_idx
                        break
                    attempts += 1
                
                if valid_candidate != -1:
                    others = [i for i in current_set_list if i != remove_idx]
                    v_rem = count_violations_with_candidate(remove_idx, others)
                    v_add = count_violations_with_candidate(valid_candidate, others)
                    
                    delta_violations = v_add - v_rem
                    delta_cost = delta_violations * PENALTY
                    # k stays same
                    
                    new_set_list = [i for i in current_set_list if i != remove_idx] + [valid_candidate]
                    new_set_set = (current_set_set - {remove_idx}) | {valid_candidate}
        
        # Acceptance Criterion
        if delta_cost < 0:
            accept = True
        else:
            # Avoid overflow in exp
            if T > 0:
                prob = math.exp(-delta_cost / T)
                accept = np.random.random() < prob
            else:
                accept = False
        
        if accept:
            current_set_list = new_set_list
            current_set_set = new_set_set
            current_violations += delta_violations
            
            # Check if we found a better valid solution
            if current_violations == 0:
                if len(current_set_list) > best_k:
                    best_k = len(current_set_list)
                    best_set = list(current_set_list)
        
        # Cooling
        T *= COOLING_RATE
        if T < T_END:
            T = T_END
    
    # Final Polish: Try to add any remaining points that don't cause violations
    if current_violations == 0:
        current_set_list = list(best_set)
        current_set_set = set(current_set_list)
        # Greedy add
        for idx in range(N):
            if idx not in current_set_set:
                v = count_violations_with_candidate(idx, current_set_list)
                if v == 0:
                    current_set_list.append(idx)
                    current_set_set.add(idx)
        if len(current_set_list) > best_k:
            best_set = current_set_list
            best_k = len(best_set)

    if not best_set:
        return np.empty((0, 3), dtype=np.int64)
        
    return grid[best_set].astype(np.int64)
