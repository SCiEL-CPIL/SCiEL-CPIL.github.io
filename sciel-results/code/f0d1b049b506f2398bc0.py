import numpy as np
import random
import time

N = 64
NUM_POINTS = N * N
DEADLINE = time.time() + 290

# Pre-compute coordinates
points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)

# Pre-compute full distance matrix
def _precompute_distances():
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=2).astype(np.int32)

D_grid = _precompute_distances()

def is_valid_set(S_indices):
    """
    Check if the set of indices S_indices forms a no-isosceles set.
    """
    if len(S_indices) <= 2:
        return True
    
    K = len(S_indices)
    D = D_grid[np.ix_(S_indices, S_indices)]
    
    for i in range(K):
        for j in range(i + 1, K):
            d_ij = D[i, j]
            for k in range(j + 1, K):
                d_ik = D[i, k]
                d_jk = D[j, k]
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    return False
    return True

def construct():
    start_time = time.time()
    best_set = []
    all_indices = list(range(NUM_POINTS))
    
    while time.time() < DEADLINE - 10:
        random.shuffle(all_indices)
        S = []
        S_set = set()
        dist_sets = {}
        
        # Pool management: List for O(1) choice, Set for O(1) membership
        pool_list = all_indices[:]
        pool_set = set(all_indices)
        
        def is_compatible_fast(p_idx):
            if not S:
                return True
            
            d_to_S = D_grid[p_idx, S]
            
            # Check 1: uniqueness of distances from p_idx to S
            # Using np.unique is fast for small arrays
            if len(np.unique(d_to_S)) < len(S):
                return False
            
            # Check 2: d(p_idx, s_idx) not in dist_sets[s_idx]
            for i, s_idx in enumerate(S):
                if d_to_S[i] in dist_sets[s_idx]:
                    return False
            
            return True

        def add_point(p_idx):
            S.append(p_idx)
            S_set.add(p_idx)
            
            # Remove from pool
            if p_idx in pool_set:
                pool_set.remove(p_idx)
                # Remove from pool_list efficiently?
                # For now, we can leave it in pool_list and filter when choosing?
                # Or use a different structure.
                # Since pool_list is large, removing is O(N).
                # Let's just mark it as removed?
                # Actually, we can just use a set for pool and pick random from it?
                # random.choice(set) is not O(1) in Python.
                # Let's stick to list and accept O(N) removal, or just filter.
                # Better: Just use a set for pool, and maintain a list of valid candidates?
                # Simpler: Use a list, and when choosing, if index is not in pool_set, retry.
                pass 

            new_dist_set = set()
            for s_idx in S[:-1]:
                d = D_grid[p_idx, s_idx]
                dist_sets[s_idx].add(d)
                new_dist_set.add(d)
            dist_sets[p_idx] = new_dist_set

        def remove_point(p_idx):
            S.remove(p_idx)
            S_set.remove(p_idx)
            
            # Add to pool
            pool_set.add(p_idx)
            pool_list.append(p_idx) # Just append, handle duplicates in choice

            p_dist_set = dist_sets.pop(p_idx)
            for s_idx in S:
                d = D_grid[p_idx, s_idx]
                dist_sets[s_idx].discard(d)

        def get_random_pool_point():
            while True:
                if not pool_list:
                    return None
                idx = random.choice(pool_list)
                if idx in pool_set:
                    return idx
                else:
                    # Clean up pool_list occasionally?
                    # For now, just retry.
                    continue

        # Greedy Phase
        for idx in all_indices:
            if is_compatible_fast(idx):
                add_point(idx)
        
        # Local Search Phase
        local_search_deadline = time.time() + 5
        while time.time() < local_search_deadline:
            if not pool_set:
                break
            
            p_idx = get_random_pool_point()
            if p_idx is None:
                break
            
            if is_compatible_fast(p_idx):
                add_point(p_idx)
                if len(S) > len(best_set):
                    best_set = S[:]
            else:
                if S:
                    rem_idx = random.choice(S)
                    remove_point(rem_idx)
                    
                    # Try to add p_idx
                    if is_compatible_fast(p_idx):
                        add_point(p_idx)
                        # Try to add one more
                        p2_idx = get_random_pool_point()
                        if p2_idx is not None and p2_idx != p_idx:
                            if is_compatible_fast(p2_idx):
                                add_point(p2_idx)
                    else:
                        # Try to add a different point
                        p2_idx = get_random_pool_point()
                        if p2_idx is not None and p2_idx != p_idx:
                            if is_compatible_fast(p2_idx):
                                add_point(p2_idx)
                            else:
                                # Failed, revert
                                add_point(rem_idx)
                        else:
                            add_point(rem_idx)
        
        if len(S) > len(best_set):
            best_set = S[:]
        
        if len(best_set) > 30:
            break

    if best_set and not is_valid_set(best_set):
         return [(0, 0)]
         
    return [tuple(points[i]) for i in best_set]
