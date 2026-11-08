"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with all weight-2 vectors (baseline 220)
    base_vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    base_vectors.append(v)
    
    base_set = np.array(base_vectors, dtype=np.int64)
    best_set = base_set
    best_score = len(base_set)
    
    # Try to add weight-4 vectors (norm 2) - need pairwise dist >= 2 among all
    # If we add weight-4 vectors, max norm becomes 2, so ALL pairwise distances must be >= 2
    # This means we'd need to drop weight-2 vectors that are too close to weight-4 vectors
    # Instead, let's try a pure weight-4 approach or mixed approach
    
    # Approach: Try to find a large set of vectors all with norm <= 2, pairwise dist >= 2
    # Generate candidates: weight-2 (norm sqrt(2)) and weight-4 (norm 2) and weight-1 (norm 1)
    # But if max norm is 2, need all pairwise dist >= 2
    
    # Let's try: use only weight-4 vectors with entries +/-1, need pairwise dot <= 2
    # Greedy: add vectors one by one if they satisfy dot <= 2 with all existing
    
    # First, let's try a different structural approach:
    # Use vectors of the form with entries in {0, +/-1} and exactly 2 nonzeros (current) = 220
    # Or try to find a better set via local search
    
    # Try: start with weight-2, see if we can swap in other vectors to increase count
    # Actually, let's try a greedy approach with a larger candidate pool
    
    # Generate all integer vectors with norm squared <= 4 (norm <= 2)
    # These are: weight 0 (skip), weight 1 (22), weight 2 (220), weight 3 (1320), weight 4 with +-1 (5280)
    # That's too many. Let's be smarter.
    
    # Key insight: if all vectors have norm r, need pairwise dot <= r^2/2
    # For mixed norms, need ||v_i - v_j|| >= max(||v_i||, ||v_j||)
    
    # Let's try: maximize count with norm exactly sqrt(2) -> we have 220, proven max for this norm
    # Try norm 2: need pairwise dist >= 2, i.e., dot <= 2 for norm-2 vectors
    # Weight-4 vectors with +-1 entries: C(11,4)*16 = 5280 candidates
    # Greedily select subset with pairwise dot <= 2
    
    if time.time() < deadline:
        # Generate weight-4 candidates
        w4_candidates = []
        for support in _combinations(range(D), 4):
            for signs in product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(support, signs):
                    v[idx] = s
                w4_candidates.append(v)
        
        w4_arr = np.array(w4_candidates, dtype=np.int64)
        
        # Greedy: add vectors if dot with all existing <= 2
        selected = []
        selected_arr = []
        
        for v in w4_arr:
            if time.time() > deadline - 5:
                break
            if not selected:
                selected.append(v)
                selected_arr.append(v)
                continue
            # Check dot with all selected
            dots = selected_arr @ v
            if np.all(dots <= 2):
                selected.append(v)
                selected_arr.append(v)
        
        if len(selected) > best_score:
            best_set = np.array(selected, dtype=np.int64)
            best_score = len(selected)
    
    return best_set

def _combinations(iterable, r):
    """Simple combinations generator to avoid import overhead issues."""
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i:] = range(indices[i]+1, indices[i]+1+r-i)
        yield tuple(pool[j] for j in indices)

# EVOLVE-BLOCK-END
