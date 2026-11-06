"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 330s kill
    
    n_points = N * N
    
    def deidx(i):
        return (i // N, i % N)
    
    # Precompute squared distances
    coords = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = (diff ** 2).sum(axis=2).astype(np.int32)
    
    # Max possible squared distance in [0,N)^2
    max_dist = (N-1)**2 + (N-1)**2
    
    # Distance histogram: for each point p, dist_hist[p][d] = number of selected points at distance d from p
    # This allows O(1) lookup for "is there already a point at distance d from p?"
    # But we also need to check apex-at-existing-point cases.
    
    # Let's use a different approach: maintain for each point p, a set of distances to selected points
    # And for each pair of selected points, check if adding a new point creates isosceles.
    
    # Actually, let's use the dynamic histogram approach:
    # For a set S, we maintain:
    # 1. For each point p (not necessarily in S), dist_count[p][d] = number of points in S at distance d from p
    # 2. This lets us check "apex at p" in O(1): just check if any dist_count[p][d] >= 2
    
    # But for "apex at S_i", we need to check if d(p, S_i) equals d(S_i, S_j) for some j.
    # We can precompute for each pair (i,j) in S, the distance d(S_i, S_j).
    # Then when adding p, for each i in S, we check if d(p, S_i) is in {d(S_i, S_j) : j != i}.
    
    # Let's maintain:
    # - selected: list of selected point indices
    # - dist_to_selected[p]: array of distances from p to each selected point (recomputed on demand or cached)
    # - For each selected point i, a set of distances from i to other selected points
    
    # Actually, let's just use a clean approach with numpy vectorization:
    
    def is_valid_add(p, selected_list):
        """Check if p can be added to selected_list without forming isosceles triangle."""
        if len(selected_list) < 2:
            return True
        
        S = np.array(selected_list, dtype=np.int32)
        k = len(S)
        
        # Distances from p to all points in S
        d_p_S = dist_sq[p, S]
        
        # Check 1: Apex at p - are there duplicate distances from p to S?
        if len(np.unique(d_p_S)) < k:
            return False
        
        # Check 2: Apex at any S_i
        # For each i, check if d_p_S[i] appears in distances from S_i to other S_j
        # We can vectorize this by checking if d_p_S[i] is in the set of distances from S_i to S
        # Note: d(S_i, S_i) = 0, and d_p_S[i] > 0, so we're safe.
        
        # Build distance matrix for S
        d_S_S = dist_sq[np.ix_(S, S)]
        
        # For each i, we need to check if d_p_S[i] is in d_S_S[i, :]
        # Vectorized: for each i, check if d_p_S[i] is in d_S_S[i, :]
        # This is equivalent to: (d_S_S == d_p_S[:, np.newaxis]).any(axis=1)
        # But we need to exclude the diagonal (j=i) where d_S_S[i,i]=0
        # Since d_p_S[i] > 0, d_S_S[i,i]=0 != d_p_S[i], so no issue.
        
        mask = (d_S_S == d_p_S[:, np.newaxis])
        # Check if for any i, there exists j such that mask[i,j] is True
        # But we need j != i for it to form a triangle with distinct points
        # Actually, if j == i, then d_S_S[i,i] = 0, and d_p_S[i] > 0, so mask[i,i] = False
        # So we're safe.
        
        if mask.any(axis=1).any():
            return False
        
        return True
    
    # Greedy Phase
    random.seed(42)
    point_order = list(range(n_points))
    random.shuffle(point_order)
    
    selected = []
    selected_set = set()
    
    for p in point_order:
        if time.time() > deadline:
            break
        
        if p in selected_set:
            continue
        
        if is_valid_add(p, selected):
            selected.append(p)
            selected_set.add(p)
    
    best_size = len(selected)
    
    # Local Search Phase with remove-1-add-2 swaps
    remaining = [p for p in range(n_points) if p not in selected_set]
    
    # Precompute for faster lookups
    remaining_set = set(remaining)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Try to add a point
        if remaining:
            # Pick a few random candidates from remaining
            n_candidates = min(20, len(remaining))
            candidates = random.sample(remaining, n_candidates)
            added = False
            for p in candidates:
                if is_valid_add(p, selected):
                    selected.append(p)
                    selected_set.add(p)
                    remaining.remove(p)
                    remaining_set.remove(p)
                    best_size = len(selected)
                    added = True
                    break
            
            if added:
                continue
        
        # If no point added, try remove-1-add-2 swap
        if len(selected) > 1 and len(remaining) >= 2:
            # Pick a random point to remove
            r_idx = random.randint(0, len(selected) - 1)
            r = selected[r_idx]
            
            # Try to find two points to add
            n_candidates = min(30, len(remaining))
            candidates = random.sample(remaining, n_candidates)
            
            found_swap = False
            for p1 in candidates:
                # Temporarily remove r
                temp_selected = selected[:r_idx] + selected[r_idx+1:]
                
                # Check if p1 can be added
                if not is_valid_add(p1, temp_selected):
                    continue
                
                # Check if p2 can be added to temp_selected + [p1]
                for p2 in candidates:
                    if p2 == p1:
                        continue
                    
                    temp_selected2 = temp_selected + [p1]
                    if is_valid_add(p2, temp_selected2):
                        # Perform swap: remove r, add p1 and p2
                        selected[r_idx] = p1
                        selected.append(p2)
                        selected_set.remove(r)
                        selected_set.add(p1)
                        selected_set.add(p2)
                        
                        remaining.remove(p1)
                        remaining.remove(p2)
                        remaining_set.remove(p1)
                        remaining_set.remove(p2)
                        remaining.append(r)
                        remaining_set.add(r)
                        
                        best_size = len(selected)
                        found_swap = True
                        break
                
                if found_swap:
                    break
        
        # Occasionally shuffle remaining to avoid bias
        if iteration % 100 == 0:
            random.shuffle(remaining)

    result = [deidx(i) for i in selected]
    return result
# EVOLVE-BLOCK-END
