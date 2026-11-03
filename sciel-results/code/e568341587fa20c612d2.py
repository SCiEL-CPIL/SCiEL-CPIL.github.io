"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _all_weight2():
    """All nonzero integer vectors of weight 2 (two non-zero entries, each ±1)."""
    vecs = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def _all_weight4_even():
    """All weight-4 vectors with even sum (from D11)."""
    vecs = []
    for idx in combinations(range(D), 4):
        for a in (1, -1):
            for b in (1, -1):
                for c in (1, -1):
                    d = 1
                    s = a + b + c + d
                    if s % 2 != 0:
                        continue
                    v = np.zeros(D, dtype=np.int64)
                    v[idx[0]] = a
                    v[idx[1]] = b
                    v[idx[2]] = c
                    v[idx[3]] = d
                    vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def construct():
    deadline = time.time() + 110  # safety margin
    
    # Phase 1: Start with all weight-2 vectors (110 vectors, all norm sqrt(2))
    # Max norm = sqrt(2), min pairwise distance = sqrt(2) (adjacent) or 2 (disjoint)
    # Actually need to check: for weight-2 vectors, ||u-v||^2 can be 0,2,4
    # We need ||u-v|| >= max_norm = sqrt(2), so ||u-v||^2 >= 2
    # So we cannot have two vectors that differ in only one coordinate (distance sqrt(2) < ... wait)
    
    # Let me reconsider. If we use only weight-2 vectors, max norm = sqrt(2).
    # ||u - v|| >= sqrt(2) means ||u-v||^2 >= 2.
    # For two weight-2 vectors, ||u-v||^2 ranges from 0 (same) to 4 (disjoint with opposite signs).
    # Adjacent (share one coordinate): ||u-v||^2 = 2 (if same sign on shared, or 4 if opposite... let me check)
    # u = [1,1,0,...], v = [1,-1,0,...]: u-v = [0,2,0,...], ||u-v||^2 = 4 >= 2 ✓
    # u = [1,1,0,...], v = [-1,1,0,...]: u-v = [2,0,0,...], ||u-v||^2 = 4 >= 2 ✓
    # u = [1,1,0,...], v = [1,0,1,...]: u-v = [0,1,-1,...], ||u-v||^2 = 2 >= 2 ✓
    # u = [1,1,0,...], v = [0,1,1,...]: same as above ✓
    # So ALL pairs of distinct weight-2 vectors have ||u-v||^2 >= 2. 
    # So all 110 weight-2 vectors are valid! Score = 110.
    
    w2 = _all_weight2()
    current = w2.copy()
    
    # Phase 2: Try to add weight-4 vectors (norm 2) to the set.
    # If we add a weight-4 vector (norm 2), max norm becomes 2.
    # Then ALL vectors must have pairwise distance >= 2, i.e., ||u-v||^2 >= 4.
    # For weight-2 vector u (norm sqrt(2)) and weight-4 vector v (norm 2):
    # ||u-v||^2 = 2 + 4 - 2*u·v = 6 - 2*u·v >= 4 => u·v <= 1
    # For two weight-4 vectors u, v: ||u-v||^2 = 8 - 2*u·v >= 4 => u·v <= 2
    # For two weight-2 vectors u, v: ||u-v||^2 >= 2, but we need >= 4 now!
    # So we can't keep all weight-2 vectors. We need to remove some.
    
    # Strategy: Build a combined set, then do greedy independent set with local search.
    # Given time constraints, let's try a structured approach:
    # 1. Generate candidate weight-4 vectors
    # 2. For each, check compatibility with current set
    # 3. Use greedy + 1-for-2 swaps
    
    w4 = _all_weight4_even()
    
    # Build conflict matrix between current (w2) and w4
    # A weight-4 vector v is compatible with weight-2 vector u if u·v <= 1
    # Since u has two nonzeros ±1, and v has four nonzeros ±1:
    # u·v can be -2,-1,0,1,2
    # We need u·v <= 1, so we exclude u·v = 2
    
    # For a given w4 vector v, which w2 vectors have u·v = 2?
    # u·v = 2 means both nonzero entries of u match the sign of v at those positions,
    # AND both positions are among v's nonzero positions.
    # So for v with support S (|S|=4), the conflicting w2 vectors are exactly
    # the C(4,2) = 6 vectors with support in S and matching signs.
    
    # Let's do a greedy: start with all w2, try to add w4 vectors one by one,
    # removing conflicting w2 vectors as needed, but only if net gain is positive.
    
    # Actually, simpler: let's just try to find a large compatible subset.
    # Given the 110s deadline, let me do a simpler approach:
    # Just return the 110 weight-2 vectors. That's the known good baseline.
    
    # But let's try to do better with a quick greedy on w4 vectors.
    # We'll try to add w4 vectors, and if adding v requires removing r w2 vectors,
    # we only add it if 1 > r (i.e., we gain net vectors). Since each w4 conflicts
    # with up to 6 w2 vectors, and we gain 1, we'd only add if conflicts < 1, i.e., 0.
    # That's too restrictive. Let me think differently.
    
    # Alternative: Use only weight-4 vectors. How many can we pack?
    # This is a kissing problem restricted to weight-4 vectors.
    # Let me try a greedy on w4 vectors with a time limit.
    
    if time.time() > deadline - 5:
        return current
    
    # Greedy on w4: sort by some heuristic, add if compatible
    # For speed, precompute dot products lazily
    
    selected = []
    selected_set = set()
    
    # Precompute supports for w2 vectors for quick conflict checking
    w2_supports = []
    for v in w2:
        nz = np.nonzero(v)[0]
        w2_supports.append((tuple(nz), v[nz].tolist()))
    
    # Greedy: try each w4 vector, add if compatible with all selected
    # To make this faster, use a different strategy: random ordering + multiple restarts
    
    best_count = len(current)
    best_vectors = current
    
    np.random.seed(42)
    
    n_w4 = len(w4)
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        order = np.random.permutation(n_w4)
        sel = []
        sel_set = set()
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            v = w4[idx]
            support = tuple(np.nonzero(v)[0])
            
            # Check compatibility with selected
            ok = True
            for sv in sel:
                dot = np.dot(v, sv)
                if dot > 2:
                    ok = False
                    break
            
            if ok:
                sel.append(v)
                sel_set.add(support)
        
        # Convert to array
        if len(sel) > best_count:
            best_count = len(sel)
            best_vectors = np.array(sel, dtype=np.int64)
    
    # Compare: weight-2 set (110) vs best weight-4 set
    if best_count > len(current):
        return best_vectors
    else:
        return current
# EVOLVE-BLOCK-END
