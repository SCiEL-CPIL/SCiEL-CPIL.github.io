"""Rewritten program for kissing number in dimension 11."""
import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Start with D_11 lattice vectors: all ±e_i ± e_j for i < j
    base_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                base_vectors.append(np.array(v, dtype=np.int64))
    
    vectors = np.array(base_vectors, dtype=np.int64)
    
    # Generate candidate pool: vectors with entries in {-2,-1,0,1,2}
    # with norm squared between 2 and 8 (to keep things manageable)
    candidates = []
    for coords in itertools.product(range(-2, 3), repeat=D):
        c = np.array(coords, dtype=np.int64)
        norm_sq = np.dot(c, c)
        if 2 <= norm_sq <= 8:
            candidates.append(c)
    
    # Remove duplicates and zero vectors
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen and 0 not in [tuple(c)]:
            seen.add(key)
            unique_candidates.append(c)
    
    # Actually, let's be more careful and efficient
    # Use a smarter candidate generation
    candidates2 = []
    for norm_sq_target in range(2, 9):
        # Generate all integer vectors with this norm squared
        # Use recursive generation
        def gen(idx, remaining, current):
            if idx == D:
                if remaining == 0:
                    candidates2.append(tuple(current))
                return
            # Try all possible values for this coordinate
            max_val = int(np.sqrt(remaining)) + 1
            for val in range(-max_val, max_val + 1):
                if val * val <= remaining:
                    current.append(val)
                    gen(idx + 1, remaining - val * val, current)
                    current.pop()
        gen(0, norm_sq_target, [])
    
    # Deduplicate
    seen2 = set()
    final_candidates = []
    for c in candidates2:
        if c not in seen2:
            seen2.add(c)
            final_candidates.append(np.array(c, dtype=np.int64))
    
    # Convert to numpy array for efficiency
    cand_array = np.array(final_candidates, dtype=np.int64)
    if len(cand_array) == 0:
        return vectors
    
    # Compute norms squared of candidates
    cand_norms_sq = np.sum(cand_array ** 2, axis=1)
    
    # Greedily add vectors that maintain the constraint
    # Constraint: for all existing v and new w: ||v - w|| >= max_norm
    # where max_norm = max of all vector norms
    # Equivalently: v·w <= (||v||² + ||w||² - max_norm²) / 2 ... this is complex
    # Simpler: just check ||v - w|| >= max_norm for all v
    
    # Start with base vectors
    vec_list = [v.copy() for v in vectors]
    max_norm_sq = max(np.dot(v, v) for v in vec_list)
    
    # Sort candidates by norm (prefer smaller norms to make constraint easier)
    cand_indices = np.argsort(cand_norms_sq)
    
    added = 0
    for idx in cand_indices:
        if time.time() > deadline:
            break
        c = cand_array[idx]
        c_norm_sq = cand_norms_sq[idx]
        
        # Check if this candidate conflicts with any existing vector
        # New max norm would be max(max_norm_sq, c_norm_sq)
        new_max_norm_sq = max(max_norm_sq, c_norm_sq)
        
        # Check: for all v in vec_list, ||v - c||² >= new_max_norm_sq
        # ||v - c||² = ||v||² + ||c||² - 2*v·c >= new_max_norm_sq
        # v·c <= (||v||² + ||c||² - new_max_norm_sq) / 2
        
        conflict = False
        for v in vec_list:
            v_norm_sq = np.dot(v, v)
            threshold = (v_norm_sq + c_norm_sq - new_max_norm_sq) / 2.0
            dot = np.dot(v, c)
            if dot > threshold + 1e-9:
                conflict = True
                break
        
        if not conflict:
            vec_list.append(c)
            max_norm_sq = new_max_norm_sq
            added += 1
    
    result = np.array(vec_list, dtype=np.int64)
    return result
# EVOLVE-BLOCK-END
