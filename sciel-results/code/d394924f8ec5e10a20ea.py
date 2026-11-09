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
    deadline = start_time + 590  # Safety margin before 630s kill
    
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
    
    # Precompute: for each element, list of non-conflicting elements (sorted)
    # nonconflict[i] = list of j such that i does NOT conflict with j, j != i
    nonconflict = [[] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i != j and (conflict_mask[i] & (1 << j)) == 0:
                nonconflict[i].append(j)

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

    def greedy_fill(mask, exclude_mask=0):
        """Greedily add all compatible elements not in mask or exclude_mask."""
        new_mask = mask
        for v in range(m):
            bit = 1 << v
            if (new_mask & bit) == 0 and (exclude_mask & bit) == 0 and (conflict_mask[v] & new_mask) == 0:
                new_mask |= bit
        return new_mask

    def local_search(subset, mask, max_iter=50):
        """
        Local search: try to increase size via remove+fill and swap moves.
        Returns improved subset and mask.
        """
        subset = list(subset)
        mask = mask
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            n = len(subset)
            
            if time.time() > deadline - 10:
                break
            
            # Try swap: replace one element with a compatible unselected one
            for i in range(n):
                if time.time() > deadline - 10:
                    return subset, mask
                u = subset[i]
                # Remove u
                trial_mask = mask & ~(1 << u)
                
                # Find a compatible element to add that isn't already selected
                best_add = -1
                for v in range(m):
                    if (trial_mask & (1 << v)) == 0 and (conflict_mask[v] & trial_mask) == 0:
                        best_add = v
                        break
                
                if best_add >= 0:
                    # Check if this actually increases size (it should if best_add != u)
                    if best_add != u:
                        new_mask = trial_mask | (1 << best_add)
                        new_subset = [x for x in subset if x != u] + [best_add]
                        # Verify no conflicts
                        valid = True
                        for x in new_subset:
                            if (conflict_mask[x] & new_mask) & ~(1 << x):
                                # Check pairwise
                                pass
                        # Quick check: just verify the new element doesn't conflict with others
                        if (conflict_mask[best_add] & trial_mask) == 0:
                            subset = new_subset
                            mask = new_mask
                            improved = True
                            break
            
            # Try removing 1 element and greedily filling
            if not improved:
                for i in range(n):
                    if time.time() > deadline - 10:
                        return subset, mask
                    u = subset[i]
                    # Remove u
                    trial_mask = mask & ~(1 << u)
                    
                    # Greedily add back compatible elements
                    new_mask = greedy_fill(trial_mask)
                    
                    new_subset = [x for x in range(m) if (new_mask & (1 << x)) != 0]
                    
                    if len(new_subset) > len(subset):
                        subset = new_subset
                        mask = new_mask
                        improved = True
                        break
            
            # Try removing 2 elements and greedily filling
            if not improved:
                for i in range(n):
                    if time.time() > deadline - 10:
                        return subset, mask
                    for j in range(i+1, n):
                        if time.time() > deadline - 10:
                            return subset, mask
                        u, v = subset[i], subset[j]
                        # Remove u and v
                        trial_mask = mask & ~(1 << u) & ~(1 << v)
                        
                        # Greedily add back compatible elements
                        new_mask = greedy_fill(trial_mask)
                        new_subset = [x for x in range(m) if (new_mask & (1 << x)) != 0]
                        
                        if len(new_subset) > len(subset):
                            subset = new_subset
                            mask = new_mask
                            improved = True
                            break
                    if improved:
                        break
        
        return subset, mask

    def polish(subset, mask):
        """Final high-precision polish: try all single swaps and small removals."""
        subset = list(subset)
        mask = mask
        improved = True
        while improved:
            improved = False
            n = len(subset)
            
            if time.time() > deadline - 5:
                break
            
            # Try all single swaps
            for i in range(n):
                if time.time() > deadline - 5:
                    return subset, mask
                u = subset[i]
                trial_mask = mask & ~(1 << u)
                
                # Find best compatible replacement
                for v in range(m):
                    if v == u:
                        continue
                    if (trial_mask & (1 << v)) == 0 and (conflict_mask[v] & trial_mask) == 0:
                        new_mask = trial_mask | (1 << v)
                        new_subset = [x for x in subset if x != u] + [v]
                        subset = new_subset
                        mask = new_mask
                        improved = True
                        break
                if improved:
                    break
            
            # Try removing 1 and filling
            if not improved:
                for i in range(n):
                    if time.time() > deadline - 5:
                        return subset, mask
                    u = subset[i]
                    trial_mask = mask & ~(1 << u)
                    new_mask = greedy_fill(trial_mask)
                    new_subset = [x for x in range(m) if (new_mask & (1 << x)) != 0]
                    if len(new_subset) > len(subset):
                        subset = new_subset
                        mask = new_mask
                        improved = True
                        break
        
        return subset, mask

    best_subset = []
    best_size = 0
    best_mask = 0
    
    random.seed(42)
    
    # Run many random greedy trials with local search
    for trial in range(20000):
        if time.time() > deadline - 15:
            break
            
        order = list(range(m))
        random.shuffle(order)
        
        subset, mask = greedy_construct(order)
        
        # Apply local search to try to improve
        subset, mask = local_search(subset, mask, max_iter=30)
        
        # Apply polish for best candidates
        if len(subset) >= best_size - 1:
            subset, mask = polish(subset, mask)
        
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
    
    # Final polish on the best result
    if time.time() < deadline - 5:
        best_subset, best_mask = polish(best_subset, best_mask)
        best_size = len(best_subset)

    return np.array(best_subset, dtype=np.int64)
