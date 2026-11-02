import itertools
import numpy as np
import time
import sys

D = 11

# EVOLVE-BLOCK-START
def _generate_weight2_candidates():
    """Generate all integer vectors of weight 2 in R^11."""
    candidates = []
    for indices in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def _generate_weight3_candidates():
    """Generate all integer vectors of weight 3 in R^11."""
    candidates = []
    for indices in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def _check_validity(vectors):
    """Check if vectors satisfy the kissing configuration condition."""
    if len(vectors) == 0:
        return False
    norms_sq = np.sum(vectors * vectors, axis=1)
    max_norm = np.max(norms_sq)
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            diff = vectors[i] - vectors[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < max_norm:
                return False
    return True

def _greedy_add_vectors(initial_vectors, candidate_pool, deadline):
    """
    Starting from initial_vectors, greedily try to add vectors from candidate_pool
    that are compatible with all vectors already in the set.
    """
    current_set = list(initial_vectors)
    n_cand = len(candidate_pool)
    
    # Precompute norms of current set
    def is_compatible(new_vec, current_set):
        new_norm_sq = np.dot(new_vec, new_vec)
        max_norm = max(new_norm_sq, max(np.dot(v, v) for v in current_set))
        for v in current_set:
            diff = new_vec - v
            dist_sq = np.dot(diff, diff)
            if dist_sq < max_norm:
                return False
        return True
    
    # Sort candidates by norm ascending (prefer smaller norms first)
    cand_norms = np.sum(candidate_pool * candidate_pool, axis=1)
    order = np.argsort(cand_norms)
    
    added = 0
    for idx in order:
        if time.time() > deadline:
            break
        vec = candidate_pool[idx]
        if is_compatible(vec, current_set):
            current_set.append(vec)
            added += 1
    
    return current_set, added

def _local_search(initial_vectors, candidate_pool, deadline):
    """
    Try to improve the set by removing one vector and adding multiple compatible ones.
    """
    current_set = list(initial_vectors)
    n_cand = len(candidate_pool)
    
    # Build index map for candidates in current set
    set_set = set()
    for v in current_set:
        set_set.add(tuple(v))
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Try removing each vector in current_set
        for i in range(len(current_set)):
            if time.time() > deadline:
                break
            
            test_set = current_set[:i] + current_set[i+1:]
            removed = current_set[i]
            
            # Try to add as many candidates as possible
            best_additions = []
            for j in range(n_cand):
                vec = candidate_pool[j]
                if tuple(vec) in set_set and j not in [k for k in range(len(current_set))]:
                    pass  # skip if already in set (simplification)
                
                # Check if vec is in test_set
                if tuple(vec) in set(tuple(v) for v in test_set):
                    continue
                
                # Check compatibility
                max_norm = max(np.dot(vec, vec), max(np.dot(v, v) for v in test_set))
                compatible = True
                for v in test_set:
                    diff = vec - v
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < max_norm:
                        compatible = False
                        break
                if compatible:
                    best_additions.append(vec)
            
            # If we can add more than we removed (1), it's an improvement
            if len(best_additions) > 1:
                # Keep only as many as we need? Or add all?
                # Add all that are mutually compatible
                new_set = test_set
                for vec in best_additions:
                    max_norm = max(np.dot(vec, vec), max(np.dot(v, v) for v in new_set))
                    compatible = True
                    for v in new_set:
                        diff = vec - v
                        dist_sq = np.dot(diff, dist_sq)
                        if dist_sq < max_norm:
                            compatible = False
                            break
                    if compatible:
                        new_set.append(vec)
                
                if len(new_set) > len(current_set):
                    current_set = new_set
                    improved = True
    
    return current_set

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with all weight-2 vectors (220 vectors)
    w2 = _generate_weight2_candidates()
    current_set = list(w2)
    
    # Generate weight-3 candidates
    w3 = _generate_weight3_candidates()
    
    # Greedily add weight-3 vectors
    current_set, _ = _greedy_add_vectors(current_set, w3, deadline)
    
    # Try local search with combined pool
    all_candidates = np.vstack([w2, w3])
    current_set = _local_search(current_set, all_candidates, deadline)
    
    # Final validation
    result = np.array(current_set, dtype=np.int64)
    
    if not _check_validity(result):
        # Fallback to weight-2 only
        return w2
    
    return result
# EVOLVE-BLOCK-END
