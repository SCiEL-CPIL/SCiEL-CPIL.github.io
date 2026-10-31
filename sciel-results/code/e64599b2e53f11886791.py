"""Program for kissing number in dimension 11: maximize k nonzero integer vectors
with min pairwise distance >= max norm.

Strategy: Use vectors with exactly two non-zero entries (±1), which gives the D11 root lattice
kissing configuration of 220 vectors. We then try to add more vectors greedily, checking
the constraint that min pairwise distance >= max norm.

The D11 construction: all vectors e_i ± e_j for i < j, with signs ±1, ±1.
Norm^2 = 2 for each. Max norm^2 = 2.
Constraint: ||u - v||^2 >= 2 for all pairs.
||u - v||^2 = ||u||^2 + ||v||^2 - 2 u·v = 4 - 2 u·v >= 2 => u·v <= 1.

For two such vectors:
- If they share no indices: u·v = 0, OK.
- If they share 1 index: u·v = ±1 (depending on signs at that index). If same sign: u·v = 1, OK. If opposite: u·v = -1, OK.
- If they share 2 indices (same support): u·v = ±1 ± 1. If both same: 2 (bad, but this is u=v, not allowed as duplicate). If one flipped: 0, OK. If both flipped: -2, OK. But u and -u are distinct vectors.

Wait, if u and v have the same support {i,j}:
u = (a, b) at positions i,j
v = (c, d) at positions i,j
u·v = a*c + b*d
Possible values: 1+1=2, 1-1=0, -1+1=0, -1-1=-2.
So u·v = 2 only when u=v (a=c, b=d). Since we don't include duplicates, this doesn't happen.
u·v = -2 when u=-v. That's fine, distance^2 = 4-2(-2)=8 >= 2.

So all 220 vectors are mutually compatible.

Can we add more vectors? Try vectors with one non-zero entry (e_i). Norm^2 = 1.
Max norm^2 would then be max(2, 1) = 2 (from the weight-2 vectors).
Constraint for e_i vs a weight-2 vector v: ||e_i - v||^2 >= 2.
If e_i is at position i, and v has non-zero at positions p, q.
If i not in {p,q}: ||e_i - v||^2 = 1 + 2 - 0 = 3 >= 2. OK.
If i = p: v has value s_p at position p. e_i has 1 at position p.
||e_i - v||^2 = ||(1-s_p) e_p + 0*e_q + ...||^2 = (1-s_p)^2 + 0 = 0 or 4.
If s_p = 1: (1-1)^2 = 0. Distance 0. BAD.
If s_p = -1: (1-(-1))^2 = 4. OK.

So e_i can only be added if for every weight-2 vector with support containing i, the sign at i is -1.
But we have both +1 and -1 at each position in our 220 vectors. So e_i conflicts with the vector that has +1 at position i and some other position.

Thus, we cannot add any weight-1 vectors.

What about weight-3 vectors? Norm^2 = 3. Max norm^2 becomes 3.
Constraint: ||u - v||^2 >= 3 for all pairs.
For two weight-3 vectors u, v: 6 - 2 u·v >= 3 => u·v <= 1.5 => u·v <= 1 (integer).
For a weight-3 vector u and weight-2 vector v: 3 + 2 - 2 u·v >= 3 => 5 - 2 u·v >= 3 => u·v <= 1.

So we need to find weight-3 vectors that have dot product <= 1 with all existing 220 weight-2 vectors, and dot product <= 1 with each other.

This is a large search space. Let's try a greedy approach: iterate through all C(11,3) = 165 weight-3 vectors (with all 2^3=8 sign combinations = 1320 total), and greedily add those that are compatible with all already-selected vectors.

Actually, let's be smarter. We'll try to add vectors one by one, checking compatibility.
"""
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    D = 11
    vectors = []

    # D11 lattice: vectors with exactly two non-zero entries, each ±1
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)

    # Convert to numpy for efficient dot products
    vecs = np.array(vectors, dtype=np.int64)
    
    # Try to add more vectors greedily
    # Generate candidate weight-3 vectors
    candidates = []
    for triple in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(triple, signs):
                v[idx] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # For each candidate, check if it's compatible with all existing vectors
    # Compatibility: ||c - v||^2 >= max_norm^2 for all v in vectors
    # max_norm^2 = 2 (from weight-2 vectors)
    # ||c - v||^2 = ||c||^2 + ||v||^2 - 2 c·v = 3 + 2 - 2 c·v = 5 - 2 c·v >= 2 => c·v <= 1.5 => c·v <= 1
    
    # So we need c·v <= 1 for all v in vectors, and c·c' <= 1 for all other added candidates c'
    
    added = []
    for c in candidates:
        if time.time() > deadline:
            break
        
        # Check against all existing vectors
        dots = vecs @ c
        if np.any(dots > 1):
            continue
        
        # Check against all previously added candidates
        if added:
            added_arr = np.array(added, dtype=np.int64)
            dots_added = added_arr @ c
            if np.any(dots_added > 1):
                continue
        
        added.append(c)
    
    if added:
        vecs = np.vstack([vecs, np.array(added, dtype=np.int64)])
    
    return vecs
