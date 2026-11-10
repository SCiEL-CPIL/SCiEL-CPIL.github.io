import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Select as many points as possible from the 8x8x8 integer grid {0,...,7}^3
    such that no five selected points are co-spherical (or co-planar).
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    r2 = (grid ** 2).sum(axis=1)
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_set = []
    best_len = 0
    
    def can_add(p_idx, S_indices):
        """
        Check if p_idx can be added to the set defined by S_indices.
        S_indices is a list of indices into the grid.
        Uses early exit: stops as soon as one violating 4-tuple is found.
        """
        m = len(S_indices)
        if m < 4:
            return True
        
        p = grid[p_idx]
        rp2 = r2[p_idx]
        
        # Iterate over all 4-combinations of S_indices
        for combo in itertools.combinations(range(m), 4):
            s_idxs = [S_indices[i] for i in combo]
            qs = grid[s_idxs]  # (4, 3)
            qr2s = r2[s_idxs]  # (4,)
            
            # Build 4x4 matrix
            mat = np.empty((4, 4), dtype=np.float64)
            mat[:, :3] = qs - p
            mat[:, 3] = qr2s - rp2
            
            det_val = np.linalg.det(mat)
            if abs(det_val) < 0.5:
                return False
        
        return True

    def try_add_point(p_idx, S_indices):
        """Try to add p_idx to S_indices, return True if successful."""
        if can_add(p_idx, S_indices):
            S_indices.append(p_idx)
            return True
        return False

    def greedy_construct(order):
        """Greedy construction following the given order."""
        S = []
        for p_idx in order:
            if can_add(p_idx, S):
                S.append(p_idx)
        return S

    def local_refinement(S_indices):
        """
        Try to improve the set by:
        1. Trying to add unselected points
        2. Swapping selected points with unselected ones
        """
        S = list(S_indices)
        S_set = set(S)
        unselected = [i for i in range(N) if i not in S_set]
        
        improved = True
        while improved:
            improved = False
            
            # Try adding any unselected point
            for u in unselected[:]:
                if can_add(u, S):
                    S.append(u)
                    S_set.add(u)
                    unselected.remove(u)
                    improved = True
            
            # Try swapping: remove one selected, try to add one unselected
            if not improved:
                for s_idx in range(len(S)):
                    s = S[s_idx]
                    S_temp = S[:s_idx] + S[s_idx+1:]
                    for u in unselected[:]:
                        if can_add(u, S_temp):
                            # Check if we can still maintain size or improve
                            S = S_temp + [u]
                            S_set.discard(s)
                            S_set.add(u)
                            unselected.remove(u)
                            improved = True
                            break
                    if improved:
                        break
        
        return S

    np.random.seed(0)
    
    # Phase 1: Multiple random greedy constructions
    num_random_restarts = 200
    for restart in range(num_random_restarts):
        if time.time() >= deadline:
            break
        
        # Random permutation
        order = np.random.permutation(N)
        
        S_indices = greedy_construct(order)
        
        if len(S_indices) > best_len:
            best_len = len(S_indices)
            best_set = S_indices.copy()
    
    # Phase 2: Try structured orderings
    if time.time() < deadline - 60:
        # Try ordering by various criteria
        # Sort by distance from origin
        dist_from_origin = r2.copy()
        order_by_dist = np.argsort(dist_from_origin)
        S_indices = greedy_construct(order_by_dist)
        if len(S_indices) > best_len:
            best_len = len(S_indices)
            best_set = S_indices.copy()
        
        # Sort by negative distance from origin
        order_by_neg_dist = np.argsort(-dist_from_origin)
        S_indices = greedy_construct(order_by_neg_dist)
        if len(S_indices) > best_len:
            best_len = len(S_indices)
            best_set = S_indices.copy()
        
        # Try a few more random orders
        for _ in range(50):
            if time.time() >= deadline - 60:
                break
            order = np.random.permutation(N)
            S_indices = greedy_construct(order)
            if len(S_indices) > best_len:
                best_len = len(S_indices)
                best_set = S_indices.copy()
    
    # Phase 3: Local refinement on the best set found
    if best_set and time.time() < deadline - 30:
        refined = local_refinement(best_set)
        if len(refined) > best_len:
            best_len = len(refined)
            best_set = refined
    
    # Phase 4: If we have time, try more refinements with different seeds
    if time.time() < deadline - 10:
        for extra_restart in range(20):
            if time.time() >= deadline - 5:
                break
            
            # Start from a random subset of the best set and rebuild
            if best_set:
                # Take a random subset of size ~80% of best
                subset_size = max(4, int(len(best_set) * 0.8))
                subset = list(np.random.choice(best_set, size=min(subset_size, len(best_set)), replace=False))
                
                # Try to extend this subset
                S = list(subset)
                remaining = [i for i in range(N) if i not in set(S)]
                # Try adding from remaining in random order
                np.random.shuffle(remaining)
                for p_idx in remaining:
                    if time.time() >= deadline - 5:
                        break
                    if can_add(p_idx, S):
                        S.append(p_idx)
                
                if len(S) > best_len:
                    best_len = len(S)
                    best_set = S
    
    if not best_set:
        # Fallback: return a small valid set
        return grid[:4].astype(np.int64)
        
    return grid[best_set].astype(np.int64)
