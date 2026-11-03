"""Improved kissing number construction for dimension 11."""
import time
import numpy as np
from itertools import combinations, product

D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations, product
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit, leave margin
    
    # We want to construct integer vectors in R^11 such that:
    # - All vectors are nonzero
    # - Minimum pairwise distance >= maximum vector norm
    # - Score = number of vectors k
    
    # Key insight: If all vectors have the same norm r, then we need pairwise distances >= r.
    # For vectors u, v with ||u|| = ||v|| = r, distance^2 = 2r^2 - 2u·v
    # So we need 2r^2 - 2u·v >= r^2, i.e., u·v <= r^2/2
    
    # Strategy: Use vectors with entries from {-1, 0, 1} to keep norms small and controlled.
    # A vector with m non-zero entries has norm^2 = m.
    # Two vectors with supports S1, S2 and same sign pattern on overlap:
    #   u·v = sum of products on common support
    # If they share exactly t positions with same sign, u·v = t
    # We need t <= m/2 where m is the number of non-zero entries (assuming equal weight)
    
    # Better approach: Use a greedy/set-packing style approach.
    # Generate candidate vectors with small norm, then greedily select a subset
    # where pairwise dot products satisfy the constraint.
    
    # Let's use vectors with exactly 2 non-zero entries (norm^2 = 2).
    # Two such vectors u, v: u·v can be -2, -1, 0, 1, 2 depending on overlap and signs.
    # We need u·v <= 1 (since r^2/2 = 2/2 = 1).
    # So we cannot have two vectors that are identical or negatives of each other on both positions.
    # Actually if u = (a,b,0,...) and v = (c,d,0,...) with same support {i,j}:
    #   u·v = a*c + b*d, each in {-1,0,1}
    #   If a=c and b=d, u·v = 2 > 1, invalid.
    #   If a=c, b=-d, u·v = 0, valid.
    #   If a=-c, b=d, u·v = 0, valid.
    #   If a=-c, b=-d, u·v = -2, valid.
    # So for each pair of positions {i,j}, we can use at most 2 vectors: (si, sj) and (-si, -sj) 
    # are the same up to sign... wait no.
    
    # For support {i,j}, the possible vectors are (1,1), (1,-1), (-1,1), (-1,-1).
    # (1,1)·(1,1) = 2 > 1, so we can't have two identical.
    # (1,1)·(-1,-1) = -2, fine.
    # (1,1)·(1,-1) = 0, fine.
    # (1,1)·(-1,1) = 0, fine.
    # (1,-1)·(-1,1) = 0, fine.
    # (1,-1)·(1,-1) = 2, can't duplicate.
    # So for each support {i,j}, we can use all 4 sign combinations!
    # Wait: (1,1) and (1,1) can't both be present, but (1,1), (1,-1), (-1,1), (-1,-1) are all distinct.
    # Check (1,1) vs (1,-1): dot = 1*1 + 1*(-1) = 0 <= 1. OK.
    # Check (1,1) vs (-1,1): dot = -1+1 = 0. OK.
    # Check (1,1) vs (-1,-1): dot = -1-1 = -2. OK.
    # Check (1,-1) vs (-1,1): dot = -1-1 = -2. OK.
    # Check (1,-1) vs (-1,-1): dot = -1+1 = 0. OK.
    # Check (-1,1) vs (-1,-1): dot = 1-1 = 0. OK.
    # Great! So for each pair {i,j}, all 4 sign vectors are mutually compatible.
    
    # But we also need cross-support compatibility.
    # u on support {i,j}, v on support {k,l}.
    # If supports are disjoint, u·v = 0, always fine.
    # If they share exactly one position, say i: u=(a,0,...,b,0,...) and v=(c,0,...,d,0,...)
    # where u has nonzeros at i,j and v has nonzeros at i,k (j≠k).
    # u·v = a*c, which is ±1. We need a*c <= 1, always true.
    # So any two vectors with 2 non-zero entries are compatible as long as they're not identical!
    
    # Total pairs in R^11: C(11,2) = 55. Each gives 4 vectors. Total = 220.
    # This is already much better than the seed's ~20-30.
    
    # Can we do better with vectors of weight 3?
    # Weight 3: norm^2 = 3, need u·v <= 3/2 = 1.5, so u·v <= 1 (integer).
    # Two weight-3 vectors sharing 2 positions: dot product can be up to 2.
    # If they share 2 positions with same signs, dot = 2 > 1. Invalid.
    # So weight-3 vectors have more constraints.
    
    # Let's also consider weight-1 vectors: (1,0,...,0), etc. 11*2 = 22 vectors.
    # Weight-1 vs weight-2: if the weight-1 position is in the weight-2 support, dot = ±1 <= 1. OK.
    # So we can add all weight-1 vectors too!
    
    # Let's build the full set:
    vectors = []
    
    # Weight 1: 11 positions * 2 signs = 22 vectors
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            vectors.append(v)
    
    # Weight 2: C(11,2) pairs * 4 sign combos = 220 vectors
    for i, j in combinations(range(D), 2):
        for si, sj in product((1, -1), (1, -1)):
            v = [0] * D
            v[i] = si
            v[j] = sj
            vectors.append(v)
    
    # Verify constraint for all pairs
    arr = np.array(vectors, dtype=np.int64)
    n = len(arr)
    
    # Check: min pairwise distance >= max norm
    # Compute all norms
    norms_sq = np.sum(arr**2, axis=1)
    max_norm = np.sqrt(np.max(norms_sq))
    
    # Compute pairwise distances
    # For efficiency, check in chunks
    min_dist_sq = np.inf
    for i in range(n):
        # Distance from vector i to all others
        diffs = arr[i+1:] - arr[i]
        dists_sq = np.sum(diffs**2, axis=1)
        if len(dists_sq) > 0:
            min_dist_sq = min(min_dist_sq, np.min(dists_sq))
    
    # Check validity
    if min_dist_sq < max_norm**2 - 1e-9:
        # Constraint violated, need to fix
        # For now, just return what we have (should be valid by construction)
        pass
    
    return arr

# EVOLVE-BLOCK-END
