import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Generate candidates with squared norm 4
    candidates = []

    # Type 1: One 2, rest 0: ±2e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)

    # Compute dot product matrix M[i,j] = candidates[i] . candidates[j]
    # Constraint: for i != j, M[i,j] <= 2
    M = candidates @ candidates.T

    # Build conflict matrix: conflict[i,j] = True if M[i,j] > 2 (they conflict)
    # We want maximum independent set in the conflict graph
    conflict = (M > 2)
    # Ensure diagonal is False (no self-conflict)
    np.fill_diagonal(conflict, False)

    # The maximum independent set in the conflict graph is the same as
    # the maximum clique in the compatibility graph
    compat = ~conflict
    np.fill_diagonal(compat, False)

    best_set = None
    best_count = 0

    rng = np.random.default_rng(42)

    # Precompute degrees (number of conflicts for each vertex)
    degrees = np.sum(conflict, axis=1)

    # Greedy independent set with local search
    def greedy_is(order):
        """Greedy independent set construction given an order."""
        selected = []
        available = np.ones(N, dtype=bool)
        for idx in order:
            if not available[idx]:
                continue
            selected.append(idx)
            # Remove idx and all its conflicts from available
            available[idx] = False
            available[conflict[idx]] = False
            if not np.any(available):
                break
        return selected

    def local_search(selected_indices):
        """Improve independent set by trying to add vertices or swap."""
        selected = list(selected_indices)
        selected_set = set(selected)
        
        # Build available set: vertices not in selected and not conflicting with any selected
        available = np.ones(N, dtype=bool)
        for idx in selected:
            available[idx] = False
        for idx in selected:
            available[conflict[idx]] = False

        improved = True
        max_iter = 50
        it = 0
        while improved and it < max_iter:
            it += 1
            improved = False
            if time.time() > deadline - 5:
                break
                
            # Try to add any available vertex
            avail_list = np.where(available)[0]
            if len(avail_list) > 0:
                # Pick the one with fewest conflicts among other available vertices
                # (to leave more room for future additions)
                best_idx = None
                min_conf = float('inf')
                for idx in avail_list:
                    # Count conflicts with other available vertices
                    conf_count = np.sum(conflict[idx] & available)
                    if conf_count < min_conf:
                        min_conf = conf_count
                        best_idx = idx
                if best_idx is not None:
                    selected.append(best_idx)
                    available[best_idx] = False
                    available[conflict[best_idx]] = False
                    improved = True

            # Try single swaps: remove one selected, add one available
            if len(selected) > 0:
                for i in range(len(selected)):
                    if time.time() > deadline - 5:
                        break
                    removed = selected[i]
                    # Temporarily remove 'removed' from selected
                    selected.remove(removed)
                    
                    # Recompute available
                    temp_available = np.ones(N, dtype=bool)
                    for idx in selected:
                        temp_available[idx] = False
                    for idx in selected:
                        temp_available[conflict[idx]] = False
                    
                    avail_list = np.where(temp_available)[0]
                    if len(avail_list) > 0:
                        # Try to add a vertex that was previously blocked
                        best_idx = None
                        min_conf = float('inf')
                        for idx in avail_list:
                            conf_count = np.sum(conflict[idx] & temp_available)
                            if conf_count < min_conf:
                                min_conf = conf_count
                                best_idx = idx
                        if best_idx is not None:
                            selected.append(best_idx)
                            improved = True
                            break
                    else:
                        # No improvement, put back
                        selected.append(removed)

        return selected

    # Multiple greedy trials with different orderings
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break

        if trial == 0:
            # Lowest degree first (fewest conflicts)
            order = np.argsort(degrees, kind='stable')
        elif trial == 1:
            # Highest degree first
            order = np.argsort(-degrees, kind='stable')
        elif trial == 2:
            # Natural order
            order = np.arange(N)
        elif trial == 3:
            # Reverse natural order
            order = np.arange(N - 1, -1, -1)
        elif trial % 5 == 0:
            # Random
            order = rng.permutation(N)
        elif trial % 5 == 1:
            # Degree with noise
            noise = rng.random(N) * 0.5
            order = np.argsort(degrees + noise, kind='stable')
        elif trial % 5 == 2:
            # Reverse degree with noise
            noise = rng.random(N) * 0.5
            order = np.argsort(-degrees + noise, kind='stable')
        elif trial % 5 == 3:
            # Random weighted by inverse degree
            weights = 1.0 / (degrees + 1)
            order = rng.choice(N, size=N, replace=False, p=weights/weights.sum())
        else:
            # Another random
            order = rng.permutation(N)

        selected = greedy_is(order)
        selected_ls = local_search(selected)
        count = len(selected_ls)
        if count > best_count:
            best_count = count
            best_set = selected_ls[:]

    # Additional: try starting from different subsets
    # Start with Type 1 vectors (all mutually compatible)
    type1_indices = list(range(22))  # First 22 are Type 1
    for start_idx in [0, 5, 10, 15, 20]:
        if time.time() > deadline - 10:
            break
        # Start with a random subset of Type 1 vectors
        subset_size = min(start_idx + 1, 22)
        start_set = list(rng.choice(type1_indices, size=subset_size, replace=False))
        
        # Extend greedily
        selected = list(start_set)
        available = np.ones(N, dtype=bool)
        for idx in selected:
            available[idx] = False
        for idx in selected:
            available[conflict[idx]] = False
        
        # Continue with random order
        remaining_order = rng.permutation(N)
        for idx in remaining_order:
            if not available[idx]:
                continue
            selected.append(idx)
            available[idx] = False
            available[conflict[idx]] = False
            if not np.any(available):
                break
        
        selected_ls = local_search(selected)
        count = len(selected_ls)
        if count > best_count:
            best_count = count
            best_set = selected_ls[:]

    # If we found a set, return it
    if best_set is not None and len(best_set) > 0:
        return candidates[best_set].astype(np.int64)

    # Fallback: just the Type 1 vectors (22 vectors)
    # They are all orthogonal, so dot product 0 <= 2.
    return candidates[:22].astype(np.int64)
