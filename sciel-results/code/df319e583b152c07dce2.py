import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # 1. Generate Seed Vectors (Norm sqrt(2))
    # All vectors with exactly two non-zero entries +/-1
    seed_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                seed_vectors.append(v)
    
    current_set = seed_vectors
    min_dist_sq = 2  # Norm^2 of seed vectors is 2. Min dist between seeds is 2 (if disjoint) or 2 (if shared index? No, shared index dist^2=2).
    # Wait, if u=(1,1,0), v=(1,0,1), u-v=(0,1,-1), norm^2=2.
    # So min_dist_sq = 2.
    # Max norm sq = 2.
    # Condition: dist^2 >= max_norm_sq.
    
    def check_compatibility(candidate, current_list):
        cand_norm_sq = np.dot(candidate, candidate)
        # Max norm in set
        max_norm_sq = max(np.dot(v, v) for v in current_list) if current_list else 0
        required_dist_sq = max(cand_norm_sq, max_norm_sq)
        
        for v in current_list:
            diff = np.array(candidate) - np.array(v)
            dist_sq = np.dot(diff, diff)
            if dist_sq < required_dist_sq:
                return False
        return True

    # 2. Generate Norm sqrt(6) vectors (Sum squares 6: 2^2+1^2+1^2)
    # Try to add up to 32 of these
    norm6_candidates = []
    for i in range(D):
        for j, k in itertools.combinations([x for x in range(D) if x != i], 2):
            for si in (1, -1):
                for sj in (1, -1):
                    for sk in (1, -1):
                        v = [0] * D
                        v[i] = 2 * si
                        v[j] = sj
                        v[k] = sk
                        norm6_candidates.append(v)
    
    # Shuffle to randomize selection
    np.random.shuffle(norm6_candidates)
    
    count6_added = 0
    for v in norm6_candidates:
        if time.time() > deadline:
            break
        if count6_added >= 32:
            break
        if check_compatibility(v, current_set):
            current_set.append(v)
            count6_added += 1

    # 3. Generate Norm sqrt(8) vectors (Sum squares 8: 2^2+2^2 OR 1^2*8)
    norm8_candidates = []
    
    # Type A: Two 2s
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                norm8_candidates.append(v)
                
    # Type B: Eight 1s
    # This is large (42240). We might not need all, but let's generate.
    # To save time, we can generate on the fly or limit.
    # 42240 is small enough to generate.
    for indices in itertools.combinations(range(D), 8):
        # 2^8 = 256 sign combinations
        for signs in itertools.product([1, -1], repeat=8):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            norm8_candidates.append(v)
            
    np.random.shuffle(norm8_candidates)
    
    count8_added = 0
    target8 = 96
    for v in norm8_candidates:
        if time.time() > deadline:
            break
        if count8_added >= target8:
            break
        if check_compatibility(v, current_set):
            current_set.append(v)
            count8_added += 1
            
    # If we didn't reach 96, try more from the list?
    # The loop breaks if we hit target or deadline.
    
    # Convert to array
    if not current_set:
        return np.array(seed_vectors, dtype=np.int64)
        
    return np.array(current_set, dtype=np.int64)
