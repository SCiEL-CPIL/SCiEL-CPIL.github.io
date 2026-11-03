"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product


def construct():
    # Generate candidate vectors with squared norm = 4
    # Entries can be -2, -1, 0, 1, 2
    candidates = []
    
    # Type 1: one entry is ±2, rest 0
    for i in range(D):
        for s in (2, -2):
            v = [0] * D
            v[i] = s
            candidates.append(v)
    
    # Type 2: two entries are ±1, rest 0 (norm² = 2, not 4) - skip
    # Type 3: four entries are ±1, rest 0 (norm² = 4)
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for signs in product((1, -1), repeat=4):
                        v = [0] * D
                        v[i], v[j], v[k], v[l] = signs
                        candidates.append(v)
    
    # Type 4: one entry ±2, two entries ±1 (norm² = 4+1+1 = 6) - skip, different norm
    
    # We want all vectors to have the same norm for simplicity.
    # Let's use norm² = 4: either type 1 (one ±2) or type 3 (four ±1's)
    # But mixing different types: dot product between type1 and type3 can be 0, ±1, ±2
    # Type1: 2e_i. Type3: has entry at i? If yes, dot = ±2*±1 = ±2. If no, dot = 0.
    # So 2e_i and a type3 vector with v[i]=1 have dot=2, which is allowed (≤2).
    # 2e_i and type3 with v[i]=-1 have dot=-2, also allowed.
    
    # Let's be greedy: start with all type1 vectors, then add type3 vectors that have dot≤2 with all existing
    
    # Actually, let's just use type3 vectors (four ±1's) for a larger set
    # And check pairwise dot products
    
    # For efficiency, let's generate and filter
    type3_vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for signs in product((1, -1), repeat=4):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = signs
                        type3_vectors.append(v)
    
    # Greedily select vectors with pairwise dot product ≤ 2
    selected = []
    for v in type3_vectors:
        ok = True
        for w in selected:
            if np.dot(v, w) > 2:
                ok = False
                break
        if ok:
            selected.append(v)
    
    if len(selected) > 0:
        return np.array(selected, dtype=np.int64)
    else:
        # Fallback to original
        import itertools
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
