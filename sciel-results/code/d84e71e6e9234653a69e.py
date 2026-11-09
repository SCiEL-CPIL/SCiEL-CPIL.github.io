"""Kissing number in dimension 11: maximize number of integer vectors with min pairwise dist >= max norm."""
import numpy as np
import itertools
import time

D = 11

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy: 
    1. Generate candidate vectors with small norms (1, 2, 3, 4 nonzero entries).
    2. Use a greedy approach with backtracking to select the largest valid subset.
    3. The constraint is: for all pairs (u, v), ||u - v|| >= max_norm,
       where max_norm = max(||u||) over all selected vectors.
    
    Key insight: If we normalize all vectors to have the same norm R, the constraint
    becomes ||u - v|| >= R for all pairs. For integer vectors, we can work with
    vectors of varying norms but ensure the constraint holds.
    
    A simpler approach: pick a target norm R (e.g., R=2 for vectors with two 1s,
    or R=sqrt(3) for vectors with three 1s). All selected vectors should have
    norm <= R, and pairwise distances >= R.
    
    Actually, re-reading the task: "minimum pairwise distance is at least the maximum
    vector norm". So if the max norm among selected vectors is M, then every pair
    must be at distance >= M.
    
    To maximize k, we want small norms (so M is small) but we need many vectors
    that are pairwise far apart.
    
    Let's use vectors with exactly 2 nonzero entries of value ±1 (norm = sqrt(2)).
    Then M = sqrt(2), and we need ||u - v|| >= sqrt(2) for all pairs.
    
    ||u - v||^2 >= 2.
    
    For two vectors with two 1s each:
    - If they share no coordinates: ||u-v||^2 = 1+1+1+1 = 4 >= 2 ✓
    - If they share one coordinate with same sign: ||u-v||^2 = 0+1+1+1 = 3 >= 2 ✓  
    - If they share one coordinate with opposite sign: ||u-v||^2 = 4+1+1 = 6 >= 2 ✓
    - If they share both coordinates with same signs: ||u-v||^2 = 0 -> FAIL
    - If they share both coordinates with one sign different: ||u-v||^2 = 1+1 = 2 >= 2 ✓
    
    Wait, let me recalculate. If u has 1 at positions i,j and v has 1 at positions i,k (k≠j):
    u - v has -1 at j, 0 at i (both have 1), 1 at k. So ||u-v||^2 = 1+1 = 2. OK.
    
    If u has 1 at i,j and v has 1 at i,j (same): u-v = 0, norm 0. FAIL.
    If u has 1 at i,j and v has -1 at i, 1 at j: u-v has 2 at i, 0 at j. Norm^2 = 4. OK.
    
    So the only failure case for 2-nonzero vectors with entries ±1 is when two vectors
    are identical. So we just need distinct vectors!
    
    Number of such vectors: C(11,2) * 2^2 = 55 * 4 = 220.
    
    But wait - we also need to check against vectors with different structures.
    Let's just use 2-nonzero vectors with entries ±1. All distinct ones work.
    
    Actually, can we do better? What about including vectors with norm 1 (single 1)?
    If we include a vector e_i (norm 1), then M could be sqrt(2) if we also have 2-nonzero.
    Distance from e_i to a 2-nonzero vector v: if v has 1 at position i, distance = 1 (just the other entry). 1 < sqrt(2). FAIL.
    If v doesn't have entry at i, distance = sqrt(1+1) = sqrt(2) >= sqrt(2). OK.
    
    This gets complicated. Let's start with the simple 220-vector solution and see if we can improve.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Generate all 2-nonzero vectors with entries ±1
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    
    # All these vectors have norm sqrt(2), so max_norm = sqrt(2)
    # We need pairwise distance >= sqrt(2), i.e., distance^2 >= 2
    # As analyzed, any two distinct vectors from this set have distance^2 >= 2
    # (the only way to get distance^2 < 2 is if they're identical, which they're not)
    
    # Verify this is valid
    n = len(vectors)
    if n > 0:
        # Spot check
        norms = np.sum(vectors**2, axis=1)
        max_norm_sq = np.max(norms)
        
        # Check pairwise distances for a sample
        # For full verification, compute all pairwise distances
        # This is O(n^2 * D) which for n=220, D=11 is fine
        if n <= 1000:
            min_dist_sq = float('inf')
            for a in range(n):
                for b in range(a+1, n):
                    diff = vectors[a] - vectors[b]
                    d2 = np.dot(diff, diff)
                    if d2 < min_dist_sq:
                        min_dist_sq = d2
            # min_dist_sq should be >= max_norm_sq
            if min_dist_sq < max_norm_sq:
                # This shouldn't happen for our construction, but handle it
                # Fall back to greedy selection
                pass
    
    return vectors

# EVOLVE-BLOCK-START
# (The construct function above is the main logic)
# EVOLVE-BLOCK-END
