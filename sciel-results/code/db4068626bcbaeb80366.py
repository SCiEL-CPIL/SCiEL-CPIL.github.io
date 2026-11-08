"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

DEADLINE = 150.0  # evaluator kills at 150s; keep safety margin
SAFETY_MARGIN = 5.0  # seconds to stop before deadline


def _min_pairwise_dist_sq(vectors):
    """Compute the minimum pairwise squared distance among vectors."""
    if len(vectors) < 2:
        return float('inf')
    # vectors is a list of arrays or a 2D array
    if isinstance(vectors, list):
        vecs = np.array(vectors, dtype=np.int64)
    else:
        vecs = vectors
    n = len(vecs)
    # Compute all pairwise distances using broadcasting in chunks to save memory
    # For n up to a few thousand, direct computation is fine
    # dist_sq[i,j] = ||v_i - v_j||^2
    # We only need the minimum off-diagonal
    min_d2 = float('inf')
    # Use a chunked approach for large n
    chunk_size = 500
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        # Compute distances from vecs[i_start:i_end] to all vecs
        diff = vecs[i_start:i_end, np.newaxis, :] - vecs[np.newaxis, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        # Mask diagonal
        for i in range(i_start, i_end):
            d2[i - i_start, i] = float('inf')
        chunk_min = np.min(d2)
        if chunk_min < min_d2:
            min_d2 = chunk_min
    return min_d2


def _max_norm_sq(vectors):
    """Compute the maximum squared norm among vectors."""
    if len(vectors) == 0:
        return 0
    if isinstance(vectors, list):
        vecs = np.array(vectors, dtype=np.int64)
    else:
        vecs = vectors
    norms_sq = np.sum(vecs ** 2, axis=1)
    return np.max(norms_sq)


def _is_valid(vectors):
    """Check if the configuration satisfies min pairwise dist >= max norm."""
    if len(vectors) == 0:
        return False
    min_d2 = _min_pairwise_dist_sq(vectors)
    max_n2 = _max_norm_sq(vectors)
    return min_d2 >= max_n2


def _generate_candidate_pool(max_weight=2, max_norm=4):
    """Generate a pool of candidate integer vectors with small norms."""
    pool = []
    # Include vectors with entries in {-max_weight, ..., max_weight}
    # But limit to those with norm^2 <= max_norm^2
    # To keep pool manageable, focus on sparse vectors
    # Vectors with 1 non-zero entry
    for i in range(D):
        for v in range(1, max_weight + 1):
            vec = [0] * D
            vec[i] = v
            pool.append(tuple(vec))
            vec[i] = -v
            pool.append(tuple(vec))
    # Vectors with 2 non-zero entries
    for i, j in combinations(range(D), 2):
        for a in range(1, max_weight + 1):
            for b in range(1, max_weight + 1):
                for sa in (1, -1):
                    for sb in (1, -1):
                        vec = [0] * D
                        vec[i] = sa * a
                        vec[j] = sb * b
                        if a * a + b * b <= max_norm * max_norm:
                            pool.append(tuple(vec))
    # Vectors with 3 non-zero entries (only if norm allows)
    for i, j, k in combinations(range(D), 3):
        for a in range(1, max_weight + 1):
            for b in range(1, max_weight + 1):
                for c in range(1, max_weight + 1):
                    if a * a + b * b + c * c > max_norm * max_norm:
                        continue
                    for sa in (1, -1):
                        for sb in (1, -1):
                            for sc in (1, -1):
                                vec = [0] * D
                                vec[i] = sa * a
                                vec[j] = sb * b
                                vec[k] = sc * c
                                pool.append(tuple(vec))
    return pool


def construct():
    start_time = time.time()
    
    # Start with the seed configuration: all vectors with exactly two non-zero entries of +/-1
    # This gives C(11,2) * 4 = 55 * 4 = 220 vectors
    # Each has norm^2 = 2, and min pairwise distance:
    # Two such vectors: distance depends on overlap
    # If they share one index: e.g., (1,0,...,1,0...) and (1,0,...,-1,0...) -> dist^2 = (1-1)^2 + (1-(-1))^2 = 4
    # If they share no index: dist^2 = 1+1+1+1 = 4
    # If they are the same: not allowed
    # So min dist^2 = 4, max norm^2 = 2. 4 >= 2, valid.
    
    # Try to improve by adding more vectors or replacing with better ones
    
    # Strategy: Greedy packing from a candidate pool
    # 1. Generate a pool of candidate vectors
    # 2. Sort by some heuristic (e.g., norm)
    # 3. Greedily add vectors that maintain the constraint
    
    # The constraint is: for all pairs (i,j), ||v_i - v_j||^2 >= max(||v||^2 for all v)
    # This is equivalent to: min pairwise dist >= max norm
    
    # Let's try a greedy approach:
    # Start with an empty set, add vectors one by one if the constraint is maintained
    # But we need to be careful about which max norm we're using
    
    # Better approach: Fix the maximum norm M, then find the largest set of vectors 
    # with norm^2 <= M^2 and pairwise distance^2 >= M^2
    
    # For the seed: M^2 = 2, we have 220 vectors
    
    # Let's try to find better configurations
    
    # First, let's try to add vectors to the seed configuration
    # The seed has max norm^2 = 2. Can we add vectors with norm^2 = 2 or 3 or 4?
    # If we add a vector with norm^2 = 3, then max norm^2 becomes 3, and we need all pairwise dist^2 >= 3
    # The seed vectors have pairwise dist^2 >= 4, so they still satisfy >= 3
    # But the new vector must have dist^2 >= 3 to all seed vectors
    
    # Let's try a more systematic approach:
    # Generate a pool, then do greedy packing for different target max norms
    
    best_vectors = []
    best_score = 0
    
    # Generate candidate pool
    # Include vectors with small norms
    pool = _generate_candidate_pool(max_weight=2, max_norm=3)
    
    # Remove duplicates
    pool = list(set(pool))
    
    # Sort by norm (ascending) for greedy packing
    def norm_sq(v):
        return sum(x * x for x in v)
    
    pool.sort(key=norm_sq)
    
    # Try greedy packing for different max norm thresholds
    # For each possible max_norm_sq value, find the largest valid set
    
    # Actually, let's just do a greedy approach:
    # Start empty, add vectors in order of increasing norm, maintaining the constraint
    
    # But we need to be smart about the order and the constraint check
    
    # Let's try: for each candidate max_norm_sq in [2, 3, 4, 5, ...], 
    # find the largest set of vectors with norm^2 <= max_norm_sq and pairwise dist^2 >= max_norm_sq
    
    # For efficiency, we'll use a greedy approach with random restarts
    
    rng = np.random.RandomState(42)
    
    for target_max_n2 in [2, 3, 4, 5, 6, 7, 8]:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Filter pool to vectors with norm^2 <= target_max_n2
        candidates = [v for v in pool if norm_sq(v) <= target_max_n2]
        
        if len(candidates) == 0:
            continue
        
        # Greedy packing with random restarts
        best_for_target = []
        
        # Try a few random orderings
        for restart in range(3):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Shuffle candidates
            shuffled = candidates[:]
            rng.shuffle(shuffled)
            
            selected = []
            selected_arr = []
            
            for v in shuffled:
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                
                # Check if adding v maintains the constraint
                # New max norm^2 = max(target_max_n2, norm_sq(v)) but we're filtering so norm_sq(v) <= target_max_n2
                # So new max norm^2 = target_max_n2 (assuming we have at least one vector with norm^2 = target_max_n2)
                # Actually, the max norm^2 is the max over selected vectors, which could be less than target_max_n2
                # Let's just check: min pairwise dist^2 >= max norm^2 of selected + v
                
                if len(selected) == 0:
                    selected.append(v)
                    selected_arr.append(np.array(v, dtype=np.int64))
                    continue
                
                # Compute min pairwise dist^2 if we add v
                # This is min(min pairwise dist^2 of selected, min dist^2 from v to selected)
                # And max norm^2 = max(max norm^2 of selected, norm_sq(v))
                
                # For efficiency, we can maintain these incrementally
                # But for simplicity, let's just check
                
                v_arr = np.array(v, dtype=np.int64)
                
                # Distances from v to all selected
                diffs = selected_arr - v_arr
                d2_to_v = np.sum(diffs ** 2, axis=1)
                min_d2_to_v = np.min(d2_to_v)
                
                # We need to track the current min pairwise dist^2 and max norm^2
                # Let's maintain these
                if not hasattr(selected, '_min_d2'):
                    selected._min_d2 = float('inf')
                    selected._max_n2 = 0
                
                new_max_n2 = max(selected._max_n2, norm_sq(v))
                
                # We need min pairwise dist^2 >= new_max_n2
                # Current min pairwise dist^2 is selected._min_d2
                # New min pairwise dist^2 = min(selected._min_d2, min_d2_to_v)
                
                if min(selected._min_d2, min_d2_to_v) >= new_max_n2:
                    selected.append(v)
                    selected_arr.append(v_arr)
                    selected._min_d2 = min(selected._min_d2, min_d2_to_v)
                    selected._max_n2 = new_max_n2
            
            if len(selected) > len(best_for_target):
                best_for_target = selected[:]
        
        if len(best_for_target) > best_score:
            best_score = len(best_for_target)
            best_vectors = best_for_target[:]
    
    # Also try the seed configuration as a baseline
    seed_vectors = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                seed_vectors.append(tuple(v))
    
    if len(seed_vectors) > best_score:
        best_vectors = seed_vectors[:]
        best_score = len(seed_vectors)
    
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        # Return whatever we have
        pass
    
    if len(best_vectors) == 0:
        # Fallback to seed
        best_vectors = seed_vectors[:]
    
    return np.array(best_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
