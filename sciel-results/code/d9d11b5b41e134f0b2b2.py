import numpy as np
import itertools

D = 11

def construct():
    vectors = []
    
    # Type 1: 2 * e_i
    # Squared norm 4.
    for i in range(D):
        v = [0] * D
        v[i] = 2
        vectors.append(v)
        v[i] = -2
        vectors.append(v)
        
    # Type 3: e_i + e_j + e_k + e_l
    # Squared norm 4.
    # We need to select a subset such that no two share 3 indices with same signs.
    # A "signature" is (frozenset(3_indices), tuple(3_signs)).
    # There are C(11,3) * 2^3 = 165 * 8 = 1320 signatures.
    # Each vector has 4 signatures.
    # We can pick at most 1 vector per signature.
    # Max vectors = 1320 / 4 = 330.
    
    used_signatures = set()
    type3_vectors = []
    
    # Generate all type 3 vectors
    candidates = []
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    # Greedy selection
    # To improve the chance of finding a large set, we can randomize or use a specific order.
    # However, deterministic is preferred for reproducibility.
    # The current order is lexicographic on indices and signs.
    
    for v in candidates:
        # Calculate signatures
        sigs = []
        nonzero_indices = [i for i in range(D) if v[i] != 0]
        # nonzero_indices has length 4
        for k in range(4):
            # Remove k-th index
            rem_indices = [nonzero_indices[j] for j in range(4) if j != k]
            rem_signs = tuple(v[rem_indices[j]] for j in range(3))
            sig = (frozenset(rem_indices), rem_signs)
            sigs.append(sig)
            
        # Check if any signature is already used
        conflict = False
        for sig in sigs:
            if sig in used_signatures:
                conflict = True
                break
        
        if not conflict:
            type3_vectors.append(v)
            for sig in sigs:
                used_signatures.add(sig)
                
    vectors.extend(type3_vectors)
    
    return np.array(vectors, dtype=np.int64)
