"""Optimized covering system search using NumPy vectorization and count-based local search."""
import time
import numpy as np
import random

_BASE = 51975
_DEADLINE = 115.0
_START = time.time()

# Precompute divisors
def _get_divisors(n):
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

_DIVS = _get_divisors(_BASE)
_DIVS_SORTED = sorted(_DIVS)
_DIVS_REVERSE = sorted(_DIVS, reverse=True)

# Precompute indices for each modulus and residue
# indices_cache[m][r] = array of indices covered by (m, r)
_indices_cache = {}
for m in _DIVS:
    _indices_cache[m] = []
    for r in range(m):
        _indices_cache[m].append(np.arange(r, _BASE, m))

def _score(classes):
    """Exact uncovered count."""
    mask = np.zeros(_BASE, dtype=np.uint8)
    for m, r in classes:
        mask[r::_BASE//m if _BASE % m == 0 else m] = 1 # This is wrong, use slicing correctly
        # Correct: mask[r::m] = 1
        # Let's just use the mask approach for safety
    # Actually, let's just sum a coverage mask
    mask = np.zeros(_BASE, dtype=np.uint8)
    for m, r in classes:
        mask[r::m] = 1
    return _BASE - np.sum(mask)

def _build_greedy(order=None):
    """Greedy construction: moduli in given order, each picks best residue."""
    if order is None:
        order = _DIVS_SORTED
    
    counts = np.zeros(_BASE, dtype=np.int32)
    classes = []
    
    for m in order:
        best_r = 0
        best_count = -1
        
        # For each residue r, count how many new positions are covered
        for r in range(m):
            idx = _indices_cache[m][r]
            count = np.sum(counts[idx] == 0)
            if count > best_count:
                best_count = count
                best_r = r
        
        classes.append([m, best_r])
        counts[_indices_cache[m][best_r]] += 1
        
    return classes

def _local_search(classes, max_rounds=100, time_limit=None):
    """Local search: for each modulus, find best residue given others fixed."""
    if time_limit is None:
        time_limit = _DEADLINE - 5
    
    classes = [c[:] for c in classes]
    n = len(classes)
    
    # Build counts array
    counts = np.zeros(_BASE, dtype=np.int32)
    for m, r in classes:
        counts[_indices_cache[m][r]] += 1
    
    uncovered = np.sum(counts == 0)
    
    for round_num in range(max_rounds):
        if time.time() > time_limit:
            break
        improved = False
        
        for i in range(n):
            m = classes[i][0]
            r_old = classes[i][1]
            
            # Find best residue for this modulus
            best_r = r_old
            best_delta = 0 # We want to minimize uncovered, so delta = new_uncovered - old_uncovered
            
            # Calculate L: elements in idx_old that have count 1 (will become 0 if we remove)
            idx_old = _indices_cache[m][r_old]
            L = np.sum(counts[idx_old] == 1)
            
            for r_new in range(m):
                if r_new == r_old:
                    continue
                
                idx_new = _indices_cache[m][r_new]
                # G: elements in idx_new that have count 0 (will become 1 if we add)
                G = np.sum(counts[idx_new] == 0)
                
                # If we change r_old to r_new:
                # Counts in idx_old decrease by 1. If they were 1, they become 0 (uncovered increases by L)
                # Counts in idx_new increase by 1. If they were 0, they become 1 (uncovered decreases by G)
                # Net change in uncovered = L - G
                delta = L - G
                
                if delta < best_delta:
                    best_delta = delta
                    best_r = r_new
            
            if best_r != r_old:
                # Apply move
                idx_old = _indices_cache[m][r_old]
                idx_new = _indices_cache[m][best_r]
                
                counts[idx_old] -= 1
                counts[idx_new] += 1
                
                uncovered += best_delta
                
                classes[i][1] = best_r
                improved = True
        
        if not improved:
            break
    
    return classes, uncovered

def construct():
    start_time = time.time()
    best_classes = None
    best_score = _BASE + 1
    
    # 1. Start with greedy in sorted order (ascending)
    classes = _build_greedy(_DIVS_SORTED)
    classes_ls, score_ls = _local_search(classes)
    if score_ls < best_score:
        best_score = score_ls
        best_classes = [c[:] for c in classes_ls]
    
    # 2. Try greedy in reverse order (descending)
    if time.time() < _DEADLINE - 20:
        classes_rev = _build_greedy(_DIVS_REVERSE)
        classes_rev_ls, score_rev_ls = _local_search(classes_rev)
        if score_rev_ls < best_score:
            best_score = score_rev_ls
            best_classes = [c[:] for c in classes_rev_ls]
    
    # 3. Random restarts with local search
    rng = random.Random(42)
    
    while time.time() < _DEADLINE - 10:
        random_order = rng.sample(_DIVS, len(_DIVS))
        classes_rand = _build_greedy(random_order)
        
        # Local search immediately
        classes_rand_ls, score_rand = _local_search(classes_rand, max_rounds=20)
        
        if score_rand < best_score:
            best_score = score_rand
            best_classes = [c[:] for c in classes_rand_ls]
            
    return best_classes
