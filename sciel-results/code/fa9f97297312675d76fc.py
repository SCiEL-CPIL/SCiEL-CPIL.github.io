import numpy as np
import itertools
import time
import random

N = 16
POINTS = np.array(list(itertools.product(range(N), repeat=2)), dtype=int)
NUM_POINTS = len(POINTS)

# Precompute squared distances between all pairs of points
# D[i, j] = squared distance between POINTS[i] and POINTS[j]
diff = POINTS[:, np.newaxis, :] - POINTS[np.newaxis, :, :]
D = np.sum(diff * diff, axis=2)

def _is_conflict(p_idx, s_indices):
    """
    Check if point p_idx conflicts with set s_indices.
    Returns True if adding p_idx to s_indices creates an isosceles triangle.
    """
    if len(s_indices) < 2:
        return False
    
    # Distances from p to all in S
    d_p = D[p_idx, s_indices]
    
    # 1. Check if any two distances from p are equal
    # This means p, A, B form an isosceles triangle with apex p.
    if len(np.unique(d_p)) < len(d_p):
        return True
        
    # 2. Check if d(p, A) == d(A, B) for any A, B in S
    # This means triangle PAB is isosceles with apex A.
    # Similarly for apex B.
    # We can check this by looking at the submatrix D[s_indices, s_indices]
    # and comparing against d_p.
    
    # For each A in S, we check if d(p, A) is in D[A, S] (excluding A itself? No, distance to self is 0, d(p,A) > 0 usually)
    # Actually, if d(p, A) == d(A, B), then A is the apex.
    # We need to check if for any A in S, d(p, A) is in the set { d(A, B) for B in S, B != A }
    
    # Vectorized approach:
    # D_sub = D[np.ix_(s_indices, s_indices)]
    # For each row i in D_sub (corresponding to A), we check if d_p[i] is in D_sub[i, :]
    # Note: D_sub[i, i] is 0. d_p[i] is > 0 (unless p==A, but p not in S).
    
    # Let's just loop over A in S for simplicity, it's small.
    for i, a_idx in enumerate(s_indices):
        d_pa = d_p[i]
        # Check if d_pa is in D[a_idx, s_indices]
        # D[a_idx, s_indices] contains distances from A to all B in S.
        # We need to exclude the distance to itself (0) if d_pa == 0, but d_pa > 0.
        # So we just check if d_pa is in the array.
        if d_pa in D[a_idx, s_indices]:
            return True
            
    return False

def _is_maximal(s_indices):
    """
    Check if set s_indices is maximal.
    Maximal means no point outside s_indices can be added without conflict.
    """
    s_set = set(s_indices)
    for p_idx in range(NUM_POINTS):
        if p_idx in s_set:
            continue
        if not _is_conflict(p_idx, s_indices):
            return False
    return True

def _construct_maximal(start_set, all_indices=None):
    """
    Greedily add points to start_set until it is maximal.
    """
    if all_indices is None:
        all_indices = list(range(NUM_POINTS))
        
    current = list(start_set)
    current_set = set(current)
    
    # To ensure we get a "small" maximal set, we should add points that are "hard" to add?
    # Or just add any valid point.
    # To minimize size, we want to add points that block the most other points?
    # For now, just iterate through all points and add if valid.
    
    changed = True
    while changed:
        changed = False
        for p_idx in all_indices:
            if p_idx in current_set:
                continue
            if not _is_conflict(p_idx, current):
                current.append(p_idx)
                current_set.add(p_idx)
                changed = True
    return current

def _local_search(s_indices, max_iters=1000):
    """
    Local search: try to remove points.
    If removing a point keeps the set maximal, remove it.
    """
    current = list(s_indices)
    current_set = set(current)
    
    for _ in range(max_iters):
        improved = False
        # Try removing each point
        # Shuffle order to avoid bias
        indices_to_check = current[:]
        random.shuffle(indices_to_check)
        
        for p_idx in indices_to_check:
            if p_idx not in current_set:
                continue
                
            # Try removing p_idx
            new_set = [x for x in current if x != p_idx]
            
            # Check if new_set is maximal
            if _is_maximal(new_set):
                current = new_set
                current_set.remove(p_idx)
                improved = True
                # Don't break, keep checking if we can remove more
            else:
                # Cannot remove
                pass
                
        if not improved:
            break
            
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_set = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Randomized Greedy Restarts
    # We generate random orderings, build a maximal set, then local search.
    
    num_restarts = 0
    while time.time() < deadline - 30:
        num_restarts += 1
        
        # Random order of points
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        
        # Greedy construction
        # We start empty and add points in this order if valid.
        # This produces a maximal set.
        current = []
        current_set = set()
        for p_idx in order:
            if not _is_conflict(p_idx, current):
                current.append(p_idx)
                current_set.add(p_idx)
                
        # The set is already maximal by construction (we tried to add all).
        # Wait, _construct_maximal adds points until no more can be added.
        # The loop above does exactly that.
        
        # Local search to minimize
        minimized = _local_search(current, max_iters=500)
        
        if len(minimized) < best_len:
            best_len = len(minimized)
            best_set = minimized
            # print(f"New best: {best_len} at restart {num_restarts}")
            
        # If we found a very small set, we might tighten the deadline for others?
        # No, just keep going.
        
    # Phase 2: More aggressive local search on the best found
    if best_set is not None and time.time() < deadline - 10:
        # Try local search with more iterations on the best
        final_best = _local_search(best_set, max_iters=5000)
        if len(final_best) < best_len:
            best_set = final_best
            best_len = best_len # update not needed if we return best_set
            
    # Convert indices back to (x, y) tuples
    if best_set is None:
        # Fallback: just return a valid maximal set
        best_set = _construct_maximal([])
        
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-START
# The code above is the core. I need to wrap it in the required block or just output the whole file.
# The prompt asks to rewrite the program from scratch.
# I will output the complete file.
# EVOLVE-BLOCK-END
