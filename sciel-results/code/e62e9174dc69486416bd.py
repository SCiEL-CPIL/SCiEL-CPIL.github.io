"""Improved program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    # Start with the seed set: all vectors with exactly two nonzero entries of +/-1
    # These are C(11,2) * 4 = 220 vectors
    # Each has norm sqrt(2), pairwise distances are either 0 (same), 2 (orthogonal supports),
    # or sqrt(2) (share one coordinate with same sign) or 2 (share one coordinate with opposite sign)
    # Wait, let me think more carefully.
    
    # Two vectors v, w with support {i,j} and {k,l}:
    # If disjoint supports: distance = sqrt(||v||^2 + ||w||^2) = sqrt(4) = 2
    # If share one coordinate: e.g., v has v[i]=a, v[j]=b; w has w[i]=c, w[k]=d
    #   distance^2 = (a-c)^2 + b^2 + d^2 = (a-c)^2 + 2
    #   If a=c: distance^2 = 2, distance = sqrt(2) < 2 = norm
    #   If a=-c: distance^2 = 4 + 2 = 6, distance = sqrt(6) > 2
    # If same support: v = (a,b,0...), w = (c,d,0...)
    #   distance^2 = (a-c)^2 + (b-d)^2
    #   Can be 0, 2, 4, 8 depending on signs
    
    # The seed has minimum pairwise distance sqrt(2) and max norm sqrt(2), so it's valid.
    # Score = 220.
    
    # Let's try to add vectors with 3 nonzero entries or other patterns.
    # A vector with 3 nonzero entries of +/-1 has norm sqrt(3).
    # For it to be compatible, distance to every existing vector must be >= sqrt(3).
    
    # Actually, let me reconsider. The task says "minimum pairwise distance is at least the maximum vector norm".
    # So if we mix vectors of different norms, the constraint is:
    #   min_{i!=j} ||v_i - v_j|| >= max_i ||v_i||
    
    # The seed uses all norm-sqrt(2) vectors with 2 nonzero entries.
    # min distance = sqrt(2) (when two vectors share a coordinate with same sign), max norm = sqrt(2). Valid.
    
    # Can we do better? The kissing number in R^11 is known to be between 440 and 569.
    # The Leech lattice gives a kissing number of 440 in dimension 24, but for dimension 11...
    # Actually, the best known kissing number in dimension 11 is 440 (from the D11 lattice or similar constructions).
    
    # Let me try a different approach: use vectors from the D11 lattice or construct explicitly.
    
    # D11 lattice: vectors in Z^11 with even sum of coordinates, or (Z+1/2)^11 with even sum.
    # The shortest vectors in D11 have norm sqrt(2) and are the e_i +/- e_j type (220 of them).
    # That's exactly our seed.
    
    # To get more, we could consider a scaled version or add vectors of different norm.
    
    # Alternative: use the fact that we can scale. If we take the 220 vectors and also add their negatives... 
    # but they're already included (si, sj ranges over all sign combos).
    
    # Let me try adding vectors of the form (1,1,1,0,...,0) with norm sqrt(3).
    # Distance from (1,1,1,0,...) to (1,1,0,...,0): sqrt(0+0+1) = 1 < sqrt(3). Bad.
    
    # What about vectors with entries in {0, 1, -1} but with more nonzeros?
    # A vector with k nonzeros of +/-1 has norm sqrt(k).
    # For two such vectors, distance^2 = sum of (diff)^2 over coordinates.
    # Each coordinate diff is in {-2, -1, 0, 1, 2}, so diff^2 in {0, 1, 4}.
    
    # This is getting complex. Let me try a greedy approach: start with the 220 seed vectors,
    # then try to add more vectors (from a larger candidate pool) that satisfy the distance constraint.
    
    # Candidate pool: all vectors with entries in {-1, 0, 1} with at most 4 nonzeros.
    # That's sum_{k=0}^{4} C(11,k) * 2^k = 1 + 22 + 220 + 1320 + 5280 = 6843 candidates.
    
    # But we need to check against all existing vectors. With 220 existing, adding one takes 220 checks.
    # Adding up to ~440 total means ~220 * 220 = 48400 distance checks, each O(11). Very fast.
    
    # Let's build the candidate pool and greedily add.
    
    # First, the seed set
    seed_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                seed_vectors.append(tuple(v))
    
    # Build candidate pool: all vectors with entries in {-1, 0, 1} with 1 to 5 nonzeros
    candidates = []
    for k in range(1, 6):
        for support in itertools.combinations(range(D), k):
            for signs in itertools.product((-1, 1), repeat=k):
                v = [0] * D
                for idx, s in zip(support, signs):
                    v[idx] = s
                candidates.append(tuple(v))
    
    # Remove duplicates and zero vector
    candidates = list(set(candidates))
    candidates = [c for c in candidates if any(x != 0 for x in c)]
    
    # Start with seed
    selected = list(seed_vectors)
    selected_set = set(seed_vectors)
    
    # For efficiency, precompute norms
    def norm_sq(v):
        return sum(x*x for x in v)
    
    def dist_sq(u, v):
        return sum((a-b)**2 for a, b in zip(u, v))
    
    # Current max norm squared
    max_norm_sq = max(norm_sq(v) for v in selected)
    
    # Greedy: try to add candidates that satisfy constraint
    # To make it more effective, sort candidates by norm (prefer smaller norms first to keep max_norm low)
    candidates.sort(key=lambda v: norm_sq(v))
    
    # Remove candidates already in selected
    candidates = [c for c in candidates if c not in selected_set]
    
    # Greedy addition
    for c in candidates:
        if time.time() > deadline - 10.0:
            break
        c_norm_sq = norm_sq(c)
        # New max norm squared would be max(max_norm_sq, c_norm_sq)
        new_max_norm_sq = max(max_norm_sq, c_norm_sq)
        # Check distance to all selected vectors
        ok = True
        for s in selected:
            d2 = dist_sq(c, s)
            if d2 < new_max_norm_sq:
                ok = False
                break
        if ok:
            selected.append(c)
            selected_set.add(c)
            max_norm_sq = new_max_norm_sq
    
    # Convert to array
    result = np.array(selected, dtype=np.int64)
    return result
# EVOLVE-BLOCK-END
