"""Improved kissing number construction using randomized greedy + local search."""
import time
import numpy as np
import random

D = 11
DEADLINE = 140.0  # safety margin before 150s kill

def _norm_sq(v):
    return int(np.dot(v, v))

def _dist_sq(a, b):
    d = a - b
    return int(np.dot(d, d))

def _is_valid_candidate(v, vectors, max_norm_sq):
    """Check if v can be added: ||v||^2 <= max_norm_sq and dist(v, u)^2 >= max_norm_sq for all u in vectors."""
    nv = _norm_sq(v)
    if nv > max_norm_sq:
        return False
    for u in vectors:
        if _dist_sq(v, u) < max_norm_sq:
            return False
    return True

def _try_add_random(vectors, max_norm_sq, rng, max_coord=4, max_attempts=200):
    """Try to add a random integer vector. Returns True if added."""
    for _ in range(max_attempts):
        # Generate random integer vector with coords in [-max_coord, max_coord]
        v = rng.randint(-max_coord, max_coord, size=D)
        if np.all(v == 0):
            continue
        if _is_valid_candidate(v, vectors, max_norm_sq):
            vectors.append(v)
            return True
    return False

def _greedy_construct(rng, max_coord=4, min_coord=1):
    """Greedy construction: start with a seed, then add random vectors."""
    vectors = []
    # Start with a simple seed
    seed = np.zeros(D, dtype=np.int64)
    seed[0] = 1
    vectors.append(seed)
    max_norm_sq = 1
    
    # Try to add vectors with increasing coordinate bounds
    for coord_bound in range(min_coord, max_coord + 1):
        attempts = 0
        while attempts < 5000:
            v = rng.randint(-coord_bound, coord_bound, size=D)
            if np.all(v == 0):
                attempts += 1
                continue
            nv = int(np.dot(v, v))
            # Check norm constraint
            if nv > max_norm_sq:
                attempts += 1
                continue
            # Check distance to all existing
            ok = True
            for u in vectors:
                if _dist_sq(v, u) < max_norm_sq:
                    ok = False
                    break
            if ok:
                vectors.append(v)
                attempts = 0  # reset counter on success
            else:
                attempts += 1
        if len(vectors) >= 200:  # diminishing returns
            break
    
    return vectors, max_norm_sq

def _local_improve(vectors, max_norm_sq, rng, time_limit=30):
    """Try to improve by swapping out vectors or adding new ones."""
    vectors = list(vectors)
    start = time.time()
    
    # Try adding more vectors with random search
    for coord_bound in range(1, 6):
        for _ in range(2000):
            if time.time() - start > time_limit:
                break
            v = rng.randint(-coord_bound, coord_bound, size=D)
            if np.all(v == 0):
                continue
            if _is_valid_candidate(v, vectors, max_norm_sq):
                vectors.append(v)
    
    # Try to swap: remove a vector and see if we can add two
    for i in range(len(vectors) - 1, -1, -1):
        if time.time() - start > time_limit:
            break
        # Try removing vector i and adding two new ones
        v_remove = vectors[i]
        vectors_temp = vectors[:i] + vectors[i+1:]
        added = 0
        for coord_bound in range(1, 5):
            for _ in range(500):
                if time.time() - start > time_limit:
                    break
                v_new = rng.randint(-coord_bound, coord_bound, size=D)
                if np.all(v_new == 0):
                    continue
                if _is_valid_candidate(v_new, vectors_temp, max_norm_sq):
                    vectors_temp.append(v_new)
                    added += 1
                    if added >= 2:
                        break
            if added >= 2:
                break
        if added >= 2:
            vectors = vectors_temp
    
    return vectors, max_norm_sq

def construct():
    start_time = time.time()
    best_vectors = None
    best_count = 0
    
    rng = random.Random(42)
    
    # Multiple random restarts with different strategies
    for restart in range(50):
        if time.time() - start_time > DEADLINE:
            break
        
        # Vary parameters per restart
        max_coord = 2 + (restart % 3)  # 2, 3, 4
        seed_val = 1 + (restart % 2)  # 1 or 2
        
        # Greedy construction
        vectors = []
        seed = np.zeros(D, dtype=np.int64)
        seed[0] = seed_val
        vectors.append(seed)
        max_norm_sq = seed_val * seed_val
        
        # Add vectors greedily
        for coord_bound in range(1, max_coord + 1):
            attempts = 0
            while attempts < 3000:
                if time.time() - start_time > DEADLINE:
                    break
                v = rng.randint(-coord_bound, coord_bound, size=D)
                if np.all(v == 0):
                    attempts += 1
                    continue
                nv = int(np.dot(v, v))
                if nv > max_norm_sq:
                    attempts += 1
                    continue
                ok = True
                for u in vectors:
                    if _dist_sq(v, u) < max_norm_sq:
                        ok = False
                        break
                if ok:
                    vectors.append(v)
                    attempts = 0
                else:
                    attempts += 1
            if time.time() - start_time > DEADLINE:
                break
        
        # Local improvement
        if time.time() - start_time < DEADLINE - 5:
            vectors, max_norm_sq = _local_improve(vectors, max_norm_sq, rng, 
                                                   time_limit=min(10, DEADLINE - (time.time() - start_time)))
        
        if len(vectors) > best_count:
            best_count = len(vectors)
            best_vectors = vectors[:]
        
        # Early stop if we're doing well
        if best_count >= 500 and time.time() - start_time > 30:
            break
    
    if best_vectors is None:
        # Fallback: return the seed's result
        import itertools
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_vectors = vectors
    
    return np.array(best_vectors, dtype=np.int64)
