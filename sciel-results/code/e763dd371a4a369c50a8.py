"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations, product

def _make_candidates(max_norm_sq=8):
    """Generate all integer vectors in R^D with 0 < ||v||^2 <= max_norm_sq."""
    candidates = []
    
    # Norm^2 = 1: unit vectors and their negatives
    for i in range(D):
        for s in (1, -1):
            v = [0]*D
            v[i] = s
            candidates.append(tuple(v))
    
    # Norm^2 = 2: pairs of +-1
    for i, j in combinations(range(D), 2):
        for si, sj in product((1,-1), (1,-1)):
            v = [0]*D
            v[i], v[j] = si, sj
            candidates.append(tuple(v))
    
    # Norm^2 = 3: triples of +-1
    for i, j, k in combinations(range(D), 3):
        for si, sj, sk in product((1,-1),(1,-1),(1,-1)):
            v = [0]*D
            v[i], v[j], v[k] = si, sj, sk
            candidates.append(tuple(v))
    
    # Norm^2 = 4: four nonzero entries with +-1
    for i, j, k, l in combinations(range(D), 4):
        for si, sj, sk, sl in product((1,-1),(1,-1),(1,-1),(1,-1)):
            v = [0]*D
            v[i], v[j], v[k], v[l] = si, sj, sk, sl
            candidates.append(tuple(v))
    
    # Also include vectors with a single coordinate of value +-2 (norm^2 = 4)
    for i in range(D):
        for s in (2, -2):
            v = [0]*D
            v[i] = s
            candidates.append(tuple(v))
    
    # Norm^2 = 5: one coordinate +-2 and one +-1, or five +-1s
    for i, j in combinations(range(D), 2):
        for s2 in (2, -2):
            for s1 in (1, -1):
                v = [0]*D
                v[i], v[j] = s2, s1
                candidates.append(tuple(v))
    for i, j, k, l, m in combinations(range(D), 5):
        for signs in product((1,-1), repeat=5):
            v = [0]*D
            v[i], v[j], v[k], v[l], v[m] = signs
            candidates.append(tuple(v))
    
    # Norm^2 = 6: two +-2 and one +-1, or three +-1s and three more +-1s (six +-1s), or one +-2 and four +-1s... 
    # Actually norm^2=6: possibilities are (2,1,1) type: one +-2, two +-1s; or six +-1s
    for i, j, k in combinations(range(D), 3):
        for s2 in (2, -2):
            for s1a, s1b in product((1,-1),(1,-1)):
                v = [0]*D
                v[i], v[j], v[k] = s2, s1a, s1b
                candidates.append(tuple(v))
    for i, j, k, l, m, n in combinations(range(D), 6):
        for signs in product((1,-1), repeat=6):
            v = [0]*D
            v[i], v[j], v[k], v[l], v[m], v[n] = signs
            candidates.append(tuple(v))
    
    # Norm^2 = 7: seven +-1s, or one +-2 and three +-1s (4+1+1+1=7? no: 4+1+1+1=7)
    for i, j, k, l, m, n, o in combinations(range(D), 7):
        for signs in product((1,-1), repeat=7):
            v = [0]*D
            v[i], v[j], v[k], v[l], v[m], v[n], v[o] = signs
            candidates.append(tuple(v))
    for i, j, k, l in combinations(range(D), 4):
        for s2 in (2, -2):
            for s1a, s1b, s1c in product((1,-1),(1,-1),(1,-1)):
                v = [0]*D
                v[i], v[j], v[k], v[l] = s2, s1a, s1b, s1c
                candidates.append(tuple(v))
    
    # Norm^2 = 8: eight +-1s, or two +-2s, or one +-2 and four +-1s (4+1+1+1+1=8)
    for i, j, k, l, m, n, o, p in combinations(range(D), 8):
        for signs in product((1,-1), repeat=8):
            v = [0]*D
            v[i], v[j], v[k], v[l], v[m], v[n], v[o], v[p] = signs
            candidates.append(tuple(v))
    for i, j in combinations(range(D), 2):
        for s2a, s2b in product((2,-2),(2,-2)):
            v = [0]*D
            v[i], v[j] = s2a, s2b
            candidates.append(tuple(v))
    for i, j, k, l, m in combinations(range(D), 5):
        for s2 in (2, -2):
            for s1a, s1b, s1c, s1d in product((1,-1),(1,-1),(1,-1),(1,-1)):
                v = [0]*D
                v[i], v[j], v[k], v[l], v[m] = s2, s1a, s1b, s1c, s1d
                candidates.append(tuple(v))
    
    return candidates

def _greedy_vectorized(candidates, deadline=140.0, n_trials=20):
    """
    Find a large subset of candidates where min pairwise distance >= max norm.
    Uses numpy for fast vectorized distance checks.
    """
    start_time = time.time()
    best_set = []
    best_size = 0
    
    n_cand = len(candidates)
    cand_arr = np.array(candidates, dtype=np.int64)
    norms_sq = np.sum(cand_arr * cand_arr, axis=1)
    
    # Sort by norm squared ascending
    sort_order = np.argsort(norms_sq, kind='stable')
    sorted_cand = cand_arr[sort_order]
    sorted_norms_sq = norms_sq[sort_order]
    
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(42 + trial)
        # Shuffle among same-norm groups, but keep overall norm ordering
        # Actually, let's just try different random perturbations
        order = np.arange(n_cand)
        rng.shuffle(order)
        
        # Process in norm-sorted order, but with random tie-breaking
        # Simpler: use the sorted order with random restarts by shuffling within norm groups
        # For simplicity, just use sorted order
        sel_mask = np.zeros(n_cand, dtype=bool)
        sel_count = 0
        
        for idx in sort_order:
            if time.time() > deadline - 2:
                break
            
            v = sorted_cand[idx]
            v_norm_sq = sorted_norms_sq[idx]
            
            if sel_count == 0:
                sel_mask[idx] = True
                sel_count += 1
                continue
            
            # Get selected vectors
            sel_vecs = sorted_cand[sel_mask]
            
            # Compute squared distances from v to all selected
            diffs = sel_vecs - v
            dists_sq = np.sum(diffs * diffs, axis=1)
            
            # Need all dists_sq >= v_norm_sq (since v has the largest norm so far)
            if np.all(dists_sq >= v_norm_sq):
                sel_mask[idx] = True
                sel_count += 1
        
        if sel_count > best_size:
            best_size = sel_count
            best_set = sorted_cand[sel_mask].tolist()
    
    return best_set

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm. Return as int64 array of shape (k, d).
    """
    deadline = 140.0
    start_time = time.time()
    
    # Generate candidates with norm^2 up to 8
    candidates = _make_candidates(max_norm_sq=8)
    
    # Use vectorized greedy
    best_set = _greedy_vectorized(candidates, deadline=deadline, n_trials=20)
    
    if not best_set:
        # Fallback: norm^2=2 pairs only
        vectors = []
        for i, j in combinations(range(D), 2):
            for si, sj in product((1,-1), (1,-1)):
                v = [0]*D
                v[i], v[j] = si, sj
                vectors.append(v)
        best_set = vectors
    
    # Verify the result satisfies the constraint
    if len(best_set) >= 2:
        arr = np.array(best_set, dtype=np.int64)
        max_norm_sq = int(np.max(np.sum(arr * arr, axis=1)))
        # Compute all pairwise distances
        n = len(arr)
        valid = True
        for i in range(n):
            if not valid:
                break
            diffs = arr[i+1:] - arr[i]
            dists_sq = np.sum(diffs * diffs, axis=1)
            if np.any(dists_sq < max_norm_sq):
                valid = False
        if not valid:
            # Fall back to seed solution
            vectors = []
            for i, j in combinations(range(D), 2):
                for si, sj in product((1,-1), (1,-1)):
                    v = [0]*D
                    v[i], v[j] = si, sj
                    vectors.append(v)
            best_set = vectors
    
    return np.array(best_set, dtype=np.int64)
# EVOLVE-BLOCK-END
