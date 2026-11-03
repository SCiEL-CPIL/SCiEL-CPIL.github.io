"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
import random
from collections import deque

D = 11

# EVOLVE-BLOCK-START


def _pairwise_min_dist_sq(vectors):
    """Return minimum pairwise squared distance for a list of vectors."""
    if len(vectors) < 2:
        return float('inf')
    # Use numpy for speed
    arr = np.array(vectors, dtype=np.int64)
    # Compute all pairwise squared distances
    # dist_sq[i,j] = |v_i - v_j|^2
    # = |v_i|^2 + |v_j|^2 - 2*v_i.v_j
    norms_sq = np.sum(arr * arr, axis=1)
    dot = arr @ arr.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot
    np.fill_diagonal(dist_sq, float('inf'))
    return int(dist_sq.min())


def _max_norm_sq(vectors):
    if not vectors:
        return 0
    arr = np.array(vectors, dtype=np.int64)
    norms_sq = np.sum(arr * arr, axis=1)
    return int(norms_sq.max())


def _is_valid(vectors):
    """Check if vectors satisfy: min pairwise dist >= max norm."""
    if not vectors:
        return True
    min_dist = _pairwise_min_dist_sq(vectors)
    max_norm = _max_norm_sq(vectors)
    return min_dist >= max_norm


def _greedy_build(candidate_pool, seed_vectors=None, max_iter=50000):
    """
    Greedy construction: start from seed_vectors (or empty), 
    add candidates from pool if they maintain the validity condition.
    """
    vectors = list(seed_vectors) if seed_vectors else []
    remaining = list(candidate_pool)
    
    for _ in range(max_iter):
        if not remaining:
            break
        # Try to add a random candidate from remaining
        # Use random order to avoid getting stuck in local patterns
        idx = random.randint(0, len(remaining) - 1)
        v = remaining[idx]
        remaining[idx] = remaining[-1]
        remaining.pop()
        
        test_vectors = vectors + [v]
        if _is_valid(test_vectors):
            vectors.append(v)
    
    return vectors


def _local_search(vectors, candidate_pool, max_iter=20000):
    """
    Local search: try to add more vectors by looking at combinations
    and trying swaps.
    """
    vectors = list(vectors)
    remaining = list(candidate_pool)
    random.shuffle(remaining)
    
    added = 0
    for _ in range(max_iter):
        if not remaining:
            break
        v = remaining.pop()
        test_vectors = vectors + [v]
        if _is_valid(test_vectors):
            vectors.append(v)
            added += 1
    
    return vectors


def _generate_candidate_vectors():
    """
    Generate a rich pool of candidate integer vectors in R^11.
    Focus on vectors with small norms to make the constraint easier.
    """
    candidates = set()
    
    # Type 1: Unit vectors (norm^2 = 1)
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = sign
            candidates.add(tuple(v))
    
    # Type 2: Two-coordinate vectors (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.add(tuple(v))
    
    # Type 3: Three-coordinate vectors (norm^2 = 3)
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.add(tuple(v))
    
    # Type 4: Vectors with one coordinate = 2 (norm^2 = 4)
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = 2 * sign
            candidates.add(tuple(v))
    
    # Type 5: Vectors with coordinates (2, 1) (norm^2 = 5)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = 2 * si, sj
                candidates.add(tuple(v))
                v2 = [0] * D
                v2[i], v2[j] = sj, 2 * si
                candidates.add(tuple(v2))
    
    # Type 6: Vectors with coordinates (2, 1, 1) (norm^2 = 6)
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = 2 * si, sj, sk
                    candidates.add(tuple(v))
    
    return list(candidates)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    random.seed(42)
    
    # Start with the known good configuration from the seed
    base_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                base_vectors.append(v)
    
    # Verify base is valid
    if not _is_valid(base_vectors):
        # Fallback: try to build from scratch
        base_vectors = []
    
    best = list(base_vectors)
    best_score = len(best)
    
    # Generate candidate pool
    pool = _generate_candidate_vectors()
    
    # Remove vectors already in base from pool
    base_set = set(tuple(v) for v in base_vectors)
    pool = [v for v in pool if tuple(v) not in base_set]
    
    # Also remove vectors that are negatives of each other if both in base
    # (they'd have distance 0 if same, but they're different vectors)
    
    # Multiple greedy attempts with different random seeds
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        
        # Shuffle pool for this trial
        trial_pool = list(pool)
        random.shuffle(trial_pool)
        
        # Try greedy from base
        result = _greedy_build(trial_pool, seed_vectors=best, max_iter=5000)
        
        if len(result) > best_score:
            best = result
            best_score = len(result)
        
        # Also try greedy from scratch
        result2 = _greedy_build(trial_pool, seed_vectors=None, max_iter=10000)
        if len(result2) > best_score:
            best = result2
            best_score = len(result2)
    
    # Final local search on best
    if time.time() < deadline - 5:
        pool2 = [v for v in pool if tuple(v) not in set(tuple(x) for x in best)]
        random.shuffle(pool2)
        best = _local_search(best, pool2, max_iter=20000)
        best_score = len(best)
    
    # Convert to numpy array
    result = np.array(best, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END

