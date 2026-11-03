"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate candidate vectors: all vectors with entries in {-1, 0, 1}
    # We'll focus on vectors with small support to maximize compatibility
    
    # A vector v is "compatible" with vector w if ||v - w|| >= max(||v||, ||w||)
    # For unit-ish vectors, this means the angle between them is sufficiently large
    
    # Let's think about this differently:
    # If we normalize to norm 2, we need pairwise distances >= 2 (the max norm)
    # For two vectors u, v with ||u|| = ||v|| = 2, distance = ||u-v|| >= 2
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 8 - 2<u,v> >= 4
    # So <u,v> <= 2
    # Since ||u||=||v||=2, cos(theta) = <u,v>/4 <= 1/2, so theta >= 60 degrees
    
    # For integer vectors, let's use vectors with entries in {-1, 0, 1}
    # and filter for compatibility
    
    # Generate all vectors with support size 1 or 2 (these are most compatible)
    candidates = []
    
    # Support size 1: e_i and -e_i for each coordinate
    for i in range(D):
        v = [0] * D
        v[i] = 1
        candidates.append(v)
        v[i] = -1
        candidates.append(v)
    
    # Support size 2: vectors with two non-zero entries, each in {-1, 1}
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Now we need to find the largest subset where all pairs are compatible
    # Two vectors u, v are compatible if ||u - v|| >= max(||u||, ||v||)
    
    # Let's precompute norms and check compatibility
    # For efficiency, we'll use a greedy approach with local search
    
    def norm_sq(v):
        return sum(x*x for x in v)
    
    def dist_sq(u, v):
        return sum((a-b)**2 for a, b in zip(u, v))
    
    def are_compatible(u, v):
        nu = norm_sq(u)
        nv = norm_sq(v)
        max_norm_sq = max(nu, nv)
        return dist_sq(u, v) >= max_norm_sq
    
    # Greedy: pick vectors in order of increasing "conflict potential"
    # A vector with more conflicts should be picked later
    
    n = len(candidates)
    if n == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    # Compute conflict count for each candidate
    conflicts = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for j in range(i+1, n):
            if not are_compatible(candidates[i], candidates[j]):
                conflicts[i] += 1
                conflicts[j] += 1
    
    # Sort by conflict count (ascending) - pick low-conflict vectors first
    order = np.argsort(conflicts)
    
    selected = []
    selected_set = set()
    
    for idx in order:
        if time.time() > deadline:
            break
        v = candidates[idx]
        vkey = tuple(v)
        if vkey in selected_set:
            continue
        # Check compatibility with all selected
        compatible = True
        for s in selected:
            if not are_compatible(v, s):
                compatible = False
                break
        if compatible:
            selected.append(v)
            selected_set.add(vkey)
    
    # Local search: try to replace one vector to enable adding more
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_size = len(selected)
        
        # Try removing one vector and see if we can add more
        for remove_idx in range(current_size):
            if time.time() > deadline:
                break
            remaining = selected[:remove_idx] + selected[remove_idx+1:]
            remaining_set = set(tuple(x) for x in remaining)
            
            # Try adding any unselected vector
            added = False
            for idx in order:
                if time.time() > deadline:
                    break
                v = candidates[idx]
                vkey = tuple(v)
                if vkey in remaining_set:
                    continue
                compatible = True
                for s in remaining:
                    if not are_compatible(v, s):
                        compatible = False
                        break
                if compatible:
                    new_selected = remaining + [v]
                    if len(new_selected) > current_size:
                        selected = new_selected
                        improved = True
                        added = True
                        break
            if added:
                break
    
    # Convert to numpy array
    if not selected:
        # Fallback: return the seed configuration
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
