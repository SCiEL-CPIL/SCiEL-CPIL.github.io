import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses bitmask-based greedy construction and effective local search.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        sq = pow(x, 2, m)
        if sq != 0:
            forbidden.add(sq)
    
    # Build conflict masks
    # conflict_mask[i] is an integer where bit j is set if i conflicts with j
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j1 = (i + d) % m
            j2 = (i - d) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        conflict_mask[i] = mask

    def greedy_construct(order):
        """Greedy construction given an order of candidates."""
        selected_mask = 0
        selected_list = []
        for candidate in order:
            # Check if candidate conflicts with any selected
            if (conflict_mask[candidate] & selected_mask) == 0:
                selected_mask |= (1 << candidate)
                selected_list.append(candidate)
        return selected_list, selected_mask

    def local_search(subset, mask):
        """
        Local search: try to increase size by removing elements and greedily adding back.
        Returns improved subset and mask.
        """
        subset = list(subset)
        mask = mask
        improved = True
        while improved:
            improved = False
            n = len(subset)
            
            # Try removing 1 element and greedily adding back
            for i in range(n):
                if time.time() > deadline - 5:
                    return subset, mask
                u = subset[i]
                # Remove u
                trial_mask = mask & ~(1 << u)
                trial_subset = subset[:i] + subset[i+1:]
                
                # Greedily add back compatible elements
                new_mask = trial_mask
                new_subset = list(trial_subset)
                for v in range(m):
                    if (new_mask & (1 << v)) == 0 and (conflict_mask[v] & new_mask) == 0:
                        new_mask |= (1 << v)
                        new_subset.append(v)
                
                if len(new_subset) > len(subset):
                    subset = new_subset
                    mask = new_mask
                    improved = True
                    break
            
            # Try removing 2 elements and greedily adding back
            if not improved:
                for i in range(n):
                    if time.time() > deadline - 5:
                        return subset, mask
                    for j in range(i+1, n):
                        if time.time() > deadline - 5:
                            return subset, mask
                        u, v = subset[i], subset[j]
                        # Remove u and v
                        trial_mask = mask & ~(1 << u) & ~(1 << v)
                        trial_subset = [x for x in subset if x != u and x != v]
                        
                        # Greedily add back compatible elements
                        new_mask = trial_mask
                        new_subset = list(trial_subset)
                        for w in range(m):
                            if (new_mask & (1 << w)) == 0 and (conflict_mask[w] & new_mask) == 0:
                                new_mask |= (1 << w)
                                new_subset.append(w)
                        
                        if len(new_subset) > len(subset):
                            subset = new_subset
                            mask = new_mask
                            improved = True
                            break
                    if improved:
                        break
        
        return subset, mask

    best_subset = []
    best_size = 0
    best_mask = 0
    
    random.seed(42)
    
    # Run many random greedy trials with local search
    for trial in range(10000):
        if time.time() > deadline - 10:
            break
            
        order = list(range(m))
        random.shuffle(order)
        
        subset, mask = greedy_construct(order)
        
        # Apply local search to try to improve
        subset, mask = local_search(subset, mask)
        
        if len(subset) > best_size:
            best_size = len(subset)
            best_subset = subset
            best_mask = mask
    
    # Final check
    if not best_subset:
        # Fallback
        subset, mask = greedy_construct(list(range(m)))
        best_subset = subset
        best_mask = mask
        best_size = len(best_subset)

    return np.array(best_subset, dtype=np.int64)
