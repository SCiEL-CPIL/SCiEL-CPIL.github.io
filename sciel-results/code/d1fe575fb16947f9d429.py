"""Program for kissing number in dimension 11: maximize k nonzero integer vectors
with min pairwise distance >= max norm.

Strategy: Use weight-4 vectors (4 non-zero entries, each ±1, norm²=4).
Constraint: ||u-v||² >= 4 => 4+4-2*u.v >= 4 => u.v <= 2.
For weight-4 ±1 vectors, u.v = sum over shared positions of (u_i * v_i).
If they share s positions, u.v can range from -s to s in steps of 2.
We need u.v <= 2. This allows:
- s=0: u.v=0, OK
- s=1: u.v=±1, OK
- s=2: u.v=-2,0,2, OK (all <= 2)
- s=3: u.v=-3,-1,1,3, only -3,-1,1 OK (3 is not)
- s=4: u.v=-4,-2,0,2,4, only -4,-2,0,2 OK (4 is not, which means u=v)

So the constraint is: no two vectors can share all 4 positions with the same signs (i.e., be identical).
But since we're generating distinct vectors, the main issue is when they share 3+ positions.
Actually, for s=3: if u.v=3, that means all 3 shared positions have the same sign. This is forbidden.
For s=4: if u.v=4, that means u=v, which can't happen since we generate distinct vectors.

So the real constraint is: no two vectors share exactly 3 positions with all matching signs.

This is equivalent to: the set of (support, sign_pattern) pairs must be such that no two have 
overlapping support of size 3 with identical signs on those positions.

Let's use a greedy approach with good ordering to maximize the count.
Total candidates: C(11,4) * 2^4 = 330 * 16 = 5280.

We'll sort candidates and greedily add if compatible.
"""
import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 150.0 - 15.0  # safety margin
    
    D = 11
    vectors = []
    
    # Generate all weight-4 vectors with ±1 entries
    all_vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            all_vectors.append(tuple(v))
    
    # To speed up compatibility checks, we'll use a set of (support, sign_tuple) for quick lookup
    # but actually we need to check pairwise dot products.
    # Let's use a smarter approach: for each new vector, check against all selected.
    # With 5280 candidates and potentially ~2000+ selected, this is O(5280 * 2000 * 11) ~ 116M ops, feasible.
    
    selected = []
    selected_arr = np.zeros((0, D), dtype=np.int64)
    
    # Sort by some heuristic to improve greedy packing
    # Vectors with more "negative" signs might be more compatible? Let's just try in natural order first.
    
    for v in all_vectors:
        if time.time() > deadline:
            break
        v_arr = np.array(v, dtype=np.int64)
        if len(selected) == 0:
            selected.append(v)
            selected_arr = v_arr.reshape(1, D)
        else:
            # Check dot products with all selected vectors
            dots = selected_arr @ v_arr
            if np.max(dots) <= 2:
                selected.append(v)
                selected_arr = np.vstack([selected_arr, v_arr.reshape(1, D)])
    
    return np.array(selected, dtype=np.int64)
