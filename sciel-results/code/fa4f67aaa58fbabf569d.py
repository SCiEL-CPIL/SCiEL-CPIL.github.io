import numpy as np
import itertools
import time

def generate_norm4_candidates(d=11):
    """
    Generate all integer vectors in Z^d with squared norm exactly 4.
    Types:
    1. One entry is +/-2, rest 0. (2*d vectors)
    2. Four entries are +/-1, rest 0. (C(d,4) * 2^4 vectors)
    """
    candidates = []
    
    # Type 1: +/-2 on one coordinate
    for i in range(d):
        for s in [2, -2]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Type 2: +/-1 on four coordinates
    for idxs in itertools.combinations(range(d), 4):
        i, j, k, l = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates.append(v)
                        
    return np.array(candidates)

def generate_norm2_candidates(d=11):
    """
    Generate all integer vectors in Z^d with squared norm exactly 2.
    """
    candidates = []
    for idxs in itertools.combinations(range(d), 2):
        i, j = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    return np.array(candidates)

def greedy_pack_single_norm(candidates, min_dist_sq, d, time_limit=500):
    """
    Greedy packing for vectors of a single norm.
    Condition: min_dist_sq >= max_norm_sq.
    Since all have same norm, max_norm_sq = norm_sq.
    So we need pairwise dist >= norm_sq.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    best_points = []
    best_size = 0
    num_restarts = 20 # More restarts for better coverage
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        indices = np.arange(len(candidates))
        np.random.shuffle(indices)
        shuffled_cands = candidates[indices]
        
        current_points = []
        current_min_dist_sq = float('inf')
        
        for v in shuffled_cands:
            if time.time() > deadline:
                break
            
            if not current_points:
                current_points.append(v)
                current_min_dist_sq = float('inf')
                continue
            
            existing = np.array(current_points)
            diffs = existing - v
            dists_sq = np.sum(diffs**2, axis=1)
            min_dist_to_new = np.min(dists_sq)
            
            # For single norm, max_norm_sq is constant = norm_sq
            # We need new_min_dist_sq >= norm_sq
            # new_min_dist_sq = min(old_min_dist_sq, min_dist_to_new)
            # Since old set was valid, old_min_dist_sq >= norm_sq.
            # So we just need min_dist_to_new >= norm_sq.
            
            if min_dist_to_new >= min_dist_sq:
                current_points.append(v)
                if min_dist_to_new < current_min_dist_sq:
                    current_min_dist_sq = min_dist_to_new
        
        if len(current_points) > best_size:
            best_size = len(current_points)
            best_points = current_points
            
    return np.array(best_points) if best_points else np.array([]).reshape(0, d)

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    # Fallback: D11 minimal vectors (Norm 2)
    fallback_points = generate_norm2_candidates(11)
    fallback_points = np.around(fallback_points).astype(np.int64)
    
    # Try Norm 4 packing
    try:
        if time.time() < deadline - 10:
            candidates_4 = generate_norm4_candidates(11)
            # Norm sq is 4. Min dist must be >= 4.
            best_4 = greedy_pack_single_norm(candidates_4, 4, 11, time_limit=400)
            
            if len(best_4) > len(fallback_points):
                # Validate just in case
                if len(best_4) >= 2:
                    norms_sq = np.sum(best_4**2, axis=1)
                    max_norm = np.max(norms_sq)
                    min_dist = float('inf')
                    # Quick check on a sample or full check if small
                    # Full check for k~300 is 45k pairs, fast
                    for i in range(len(best_4)):
                        for j in range(i+1, len(best_4)):
                            d_sq = np.sum((best_4[i] - best_4[j])**2)
                            if d_sq < min_dist:
                                min_dist = d_sq
                    if min_dist >= max_norm:
                        return np.around(best_4).astype(np.int64)
                        
    except Exception as e:
        pass
        
    return fallback_points
