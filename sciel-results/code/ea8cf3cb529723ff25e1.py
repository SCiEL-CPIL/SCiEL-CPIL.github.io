"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _is_sidon4(subset):
    """Check if a 4-element tuple has all pair sums (i<=j) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _has_sidon4(arr):
    """Check if any 4-subset of arr is Sidon. Returns True if exists."""
    for sub4 in itertools.combinations(arr, 4):
        if _is_sidon4(sub4):
            return True
    return False

def _max_sidon_size(candidates):
    """Find the largest Sidon subset size of the given 12-element set."""
    for size in range(12, 0, -1):
        for combo in itertools.combinations(candidates, size):
            n = len(combo)
            sums = set()
            ok = True
            for i in range(n):
                for j in range(i, n):
                    s = combo[i] + combo[j]
                    if s in sums:
                        ok = False
                        break
                    sums.add(s)
                if not ok:
                    break
            if ok:
                return size
    return 0

def _build_sum_collision_graph(values):
    """Build adjacency matrix where edge (i,j) exists if values[i]+values[j] collides with another pair sum.
    Higher edge weight = more sum collisions involving that pair."""
    n = len(values)
    # Count all pair sums
    sum_counts = {}
    for i in range(n):
        for j in range(i, n):
            s = values[i] + values[j]
            sum_counts[s] = sum_counts.get(s, 0) + 1
    
    # Build adjacency: edge weight = number of other pairs with same sum
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            s = values[i] + values[j]
            collision_count = sum_counts[s] - 1  # exclude self
            if collision_count > 0:
                adj[i, j] = collision_count
                adj[j, i] = collision_count
    return adj

def _spectral_select_dense(values, k=12, top_k=12):
    """Use eigenvector centrality on sum-collision graph to select top-k most 'dense' values.
    Returns sorted list of selected values."""
    n = len(values)
    if n <= k:
        return sorted(values)
    
    adj = _build_sum_collision_graph(values)
    
    # Compute eigenvector centrality (largest eigenvalue's eigenvector)
    try:
        eigvals, eigvecs = np.linalg.eigh(adj)
        # Largest eigenvalue is last
        centrality = np.abs(eigvecs[:, -1])
    except:
        # Fallback: use degree centrality
        centrality = adj.sum(axis=1)
    
    # Select top-k by centrality
    idx = np.argsort(centrality)[::-1][:k]
    selected = [values[i] for i in idx]
    return sorted(selected)

def _spectral_local_search(current_set, start_time, deadline, best_size, best_set):
    """Local search using spectral guidance to find replacements."""
    current_set = list(current_set)
    
    for iteration in range(3000):
        if time.time() > deadline:
            break
        
        # Build collision graph for current set
        adj = _build_sum_collision_graph(current_set)
        
        # Find the element with lowest centrality (least "dense")
        centrality = adj.sum(axis=1)
        min_idx = np.argmin(centrality)
        
        # Try to replace this element with a better one
        improved = False
        for attempt in range(30):
            if time.time() > deadline:
                break
            new_val = random.randint(0, 128)
            if new_val in current_set:
                continue
            
            neighbor = current_set[:]
            neighbor[min_idx] = new_val
            neighbor.sort()
            
            if not _has_sidon4(neighbor):
                ms = _max_sidon_size(neighbor)
                if ms < best_size:
                    best_size = ms
                    best_set = neighbor[:]
                    if best_size <= 3:
                        return best_set, best_size, True
                current_set = neighbor
                improved = True
                break
        
        if not improved:
            # Try random replacement
            idx = random.randint(0, 11)
            for _ in range(10):
                if time.time() > deadline:
                    break
                new_val = random.randint(0, 128)
                if new_val in current_set:
                    continue
                neighbor = current_set[:]
                neighbor[idx] = new_val
                neighbor.sort()
                if not _has_sidon4(neighbor):
                    ms = _max_sidon_size(neighbor)
                    if ms < best_size:
                        best_size = ms
                        best_set = neighbor[:]
                        if best_size <= 3:
                            return best_set, best_size, True
                    current_set = neighbor
                    break
    
    return best_set, best_size, False

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill
    
    best_set = list(range(12))
    best_size = _max_sidon_size(best_set)  # should be 4
    
    # Strategy 1: Spectral-guided search
    # Build a large candidate pool and use spectral clustering to find dense subsets
    random.seed(42)
    
    # Create a pool of candidates biased toward small values
    pool = list(range(0, 60))  # 0-59
    
    # Use spectral selection to find dense 12-subsets from the pool
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Randomly perturb the pool
        if restart > 0 and restart % 5 == 0:
            # Shift the pool
            shift = random.randint(-5, 5)
            pool = [max(0, min(128, x + shift)) for x in range(0, 60)]
            pool = list(set(pool))
            if len(pool) < 12:
                pool = list(range(0, 60))
        
        # Select top-12 by spectral centrality
        selected = _spectral_select_dense(pool, k=12, top_k=12)
        
        if not _has_sidon4(selected):
            ms = _max_sidon_size(selected)
            if ms < best_size:
                best_size = ms
                best_set = selected
                if best_size <= 3:
                    return best_set
    
    # Strategy 2: Structured sets - dense clusters
    for start in range(0, 118):
        if time.time() > deadline:
            break
        candidates = list(range(start, start + 12))
        if max(candidates) > 128:
            break
        if not _has_sidon4(candidates):
            ms = _max_sidon_size(candidates)
            if ms < best_size:
                best_size = ms
                best_set = candidates
                if best_size <= 3:
                    return best_set
    
    # Strategy 3: Arithmetic progressions
    for d in range(1, 12):
        for a in range(0, 129 - 11*d):
            if time.time() > deadline:
                break
            candidates = [a + i*d for i in range(12)]
            if max(candidates) > 128:
                continue
            if not _has_sidon4(candidates):
                ms = _max_sidon_size(candidates)
                if ms < best_size:
                    best_size = ms
                    best_set = candidates
                    if best_size <= 3:
                        return best_set
    
    # Strategy 4: Random search with fast Sidon-4 check
    for attempt in range(20000):
        if time.time() > deadline:
            break
        candidates = sorted(random.sample(range(129), 12))
        if not _has_sidon4(candidates):
            ms = _max_sidon_size(candidates)
            if ms < best_size:
                best_size = ms
                best_set = candidates
                if best_size <= 3:
                    return best_set
    
    # Strategy 5: Spectral local search
    if best_size > 3:
        best_set, best_size, done = _spectral_local_search(best_set, start_time, deadline, best_size, best_set)
        if done:
            return best_set
    
    # Strategy 6: Targeted construction
    structured_candidates = [
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 18],
        [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13],
        [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17],
        [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21],
        [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27],
    ]
    
    for cand in structured_candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) == 12 and max(cand) <= 128:
            if not _has_sidon4(cand):
                ms = _max_sidon_size(cand)
                if ms < best_size:
                    best_size = ms
                    best_set = cand[:]
                    if best_size <= 3:
                        return best_set
    
    # Strategy 7: More aggressive random search with bias toward dense sets
    for attempt in range(50000):
        if time.time() > deadline:
            break
        max_val = random.choice([20, 30, 40, 50, 80, 128])
        candidates = sorted(random.sample(range(max_val + 1), 12))
        if not _has_sidon4(candidates):
            ms = _max_sidon_size(candidates)
            if ms < best_size:
                best_size = ms
                best_set = candidates
                if best_size <= 3:
                    return best_set
    
    return best_set

# EVOLVE-BLOCK-END
