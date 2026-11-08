"""Rewritten program for kissing number in dimension 11."""
import time
import itertools
import numpy as np

D = 11
TIME_LIMIT = 115  # seconds, safety margin under 120s

# EVOLVE-BLOCK-START

def _dist_sq(v1, v2):
    diff = v1 - v2
    return int(np.dot(diff, diff))


def _check_candidate(vec, existing, min_dist_sq_threshold):
    """Check if vec can be added to existing set while maintaining min pairwise distance >= max_norm_sq."""
    # Compute max norm squared of existing
    if len(existing) == 0:
        return True
    # We need min pairwise distance >= max norm of all vectors including new one
    # For efficiency, we check against threshold = max(existing_max_norm_sq, norm_sq(vec))
    vec_norm_sq = int(np.dot(vec, vec))
    # The constraint is: for all pairs (i,j) in new set, dist_sq(i,j) >= max_norm_sq(new set)
    # Since we're adding vec, the new max_norm_sq = max(old_max, vec_norm_sq)
    # We need to check all existing vectors against vec, and also ensure existing pairs still satisfy
    # But existing pairs already satisfied dist >= old_max. If vec_norm_sq <= old_max, then old pairs still satisfy.
    # If vec_norm_sq > old_max, we need to re-check ALL existing pairs, which is expensive.
    # For now, assume we only add vectors with norm_sq <= current max, or handle the case where we need to.
    
    # Simple approach: check all existing vectors against vec
    for e in existing:
        d = _dist_sq(vec, e)
        if d < min_dist_sq_threshold:
            return False
    return True


def construct():
    start_time = time.time()
    
    # Seed: all ±e_i ± e_j for i < j (norm squared = 2)
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vectors.append(v)
    
    if len(vectors) == 0:
        return np.array(vectors, dtype=np.int64)
    
    # Compute max norm squared of seed
    max_norm_sq = max(int(np.dot(v, v)) for v in vectors)
    
    # Generate candidate pool: vectors with small norm
    # Include ±e_i (norm^2=1), ±e_i±e_j (norm^2=2), and some norm^2=3,4 vectors
    candidates = []
    
    # Norm squared 1: ±e_i
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # Norm squared 2: ±e_i ± e_j (already in seed, but add for completeness)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Norm squared 3: three indices with ±1
    for combo in itertools.combinations(range(D), 3):
        for signs in itertools.product((1, -1), repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    
    # Norm squared 4: four indices with ±1, or two indices with ±2
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    
    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # Greedy: try to add candidates with norm_sq <= max_norm_sq first
    # Then try larger norm vectors if they allow the set to remain valid
    existing = list(vectors)
    
    # Sort candidates by norm squared (ascending) to try smaller norms first
    unique_candidates.sort(key=lambda v: int(np.dot(v, v)))
    
    # Phase 1: Add vectors with norm_sq <= current max_norm_sq
    for c in unique_candidates:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        c_norm_sq = int(np.dot(c, c))
        if c_norm_sq > max_norm_sq:
            continue
        # Check if c is already in existing
        if any(np.array_equal(c, e) for e in existing):
            continue
        if _check_candidate(c, existing, max_norm_sq):
            existing.append(c)
    
    # Phase 2: Try to add vectors with larger norm, checking full constraint
    # For these, we need to verify all pairwise distances >= new_max_norm_sq
    # This is expensive, so only try a subset
    for c in unique_candidates:
        if time.time() - start_time > TIME_LIMIT - 3:
            break
        c_norm_sq = int(np.dot(c, c))
        if c_norm_sq <= max_norm_sq:
            continue
        if any(np.array_equal(c, e) for e in existing):
            continue
        
        new_max = max(max_norm_sq, c_norm_sq)
        valid = True
        for e in existing:
            d = _dist_sq(c, e)
            if d < new_max:
                valid = False
                break
        if valid:
            # Also need to check that existing pairs still satisfy dist >= new_max
            # This is the expensive part
            n = len(existing)
            for i in range(n):
                for j in range(i+1, n):
                    d = _dist_sq(existing[i], existing[j])
                    if d < new_max:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            existing.append(c)
            max_norm_sq = new_max
    
    # Phase 3: Local search - try to swap out low-norm vectors for higher-norm ones
    # This can potentially increase the total count
    improved = True
    while improved and time.time() - start_time < TIME_LIMIT - 2:
        improved = False
        # Try removing one vector and adding two or more candidates
        # Simplified: try to add one candidate by removing one conflicting vector
        for c in unique_candidates:
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            c_norm_sq = int(np.dot(c, c))
            if any(np.array_equal(c, e) for e in existing):
                continue
            
            # Find conflicts
            new_max = max(max_norm_sq, c_norm_sq)
            conflicts = []
            for idx, e in enumerate(existing):
                d = _dist_sq(c, e)
                if d < new_max:
                    conflicts.append(idx)
            
            # Also check if existing pairs would violate new_max
            pairs_ok = True
            for i in range(len(existing)):
                for j in range(i+1, len(existing)):
                    if i in conflicts or j in conflicts:
                        continue
                    d = _dist_sq(existing[i], existing[j])
                    if d < new_max:
                        pairs_ok = False
                        break
                if not pairs_ok:
                    break
            
            if not pairs_ok:
                continue
            
            # Try removing one conflict and adding c
            for conflict_idx in conflicts:
                test_existing = [e for idx, e in enumerate(existing) if idx != conflict_idx]
                test_existing.append(c)
                
                # Verify all pairs in test_existing satisfy dist >= new_max
                all_ok = True
                for i in range(len(test_existing)):
                    for j in range(i+1, len(test_existing)):
                        d = _dist_sq(test_existing[i], test_existing[j])
                        if d < new_max:
                            all_ok = False
                            break
                    if not all_ok:
                        break
                
                if all_ok:
                    existing = test_existing
                    max_norm_sq = new_max
                    improved = True
                    break
        
        # If no improvement in this pass, break
        if not improved:
            break
    
    if len(existing) == 0:
        return np.array(vectors, dtype=np.int64)
    
    return np.array(existing, dtype=np.int64)

# EVOLVE-BLOCK-END
