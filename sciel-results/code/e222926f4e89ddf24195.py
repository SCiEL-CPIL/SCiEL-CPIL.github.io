import time
import random
import numpy as np

_BASE = 51975
_DEADLINE = 118.0
_START = time.time()

# Precompute all divisors > 1 of 51975
def _get_divisors(n):
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

_DIVS = _get_divisors(_BASE)
_DIVS_SORTED = sorted(_DIVS)
_N_DIVS = len(_DIVS)

# Precompute position arrays for each (m, r)
_POS_CACHE = {}
def _get_positions(m, r):
    key = (m, r)
    if key not in _POS_CACHE:
        _POS_CACHE[key] = np.arange(r, _BASE, m)
    return _POS_CACHE[key]

# Precompute for each modulus m, array of position arrays
_MOD_POS = {}
for m in _DIVS_SORTED:
    _MOD_POS[m] = [_get_positions(m, r) for r in range(m)]

def _greedy_build(order=None):
    """Greedy: add moduli in given order, each picks best residue."""
    if order is None:
        order = _DIVS_SORTED
    
    covered = np.zeros(_BASE, dtype=np.uint8)
    classes = []
    
    for m in order:
        best_r = 0
        best_count = -1
        pos_arrays = _MOD_POS[m]
        for r in range(m):
            pos = pos_arrays[r]
            count = np.sum(covered[pos] == 0)
            if count > best_count:
                best_count = count
                best_r = r
            if best_count == 0:
                break
        
        classes.append([m, best_r])
        pos = pos_arrays[best_r]
        covered[pos] = 1
    
    uncovered = int(np.sum(covered == 0))
    return classes, uncovered

def _local_refine(classes, max_time=None):
    """Local refinement: try changing residue of each class."""
    classes = [c[:] for c in classes]
    n_classes = len(classes)
    
    # Build coverage count array
    cov = np.zeros(_BASE, dtype=np.int32)
    for m, r in classes:
        pos = _get_positions(m, r)
        cov[pos] += 1
    
    uncovered = int(np.sum(cov == 0))
    
    if max_time is None:
        max_time = time.time() - _START + 10
    
    improved = True
    while improved:
        if time.time() > max_time:
            break
        improved = False
        
        for i in range(n_classes):
            m, r = classes[i]
            pos_old = _get_positions(m, r)
            
            best_r2 = r
            best_delta = 0
            
            pos_arrays = _MOD_POS[m]
            for r2 in range(m):
                if r2 == r:
                    continue
                pos_new = pos_arrays[r2]
                
                # Compute gain and loss efficiently
                # Positions only in new: gain if uncovered
                # Positions only in old: loss if only covered by this class
                
                # Create mask for positions in pos_new but not pos_old
                # and positions in pos_old but not pos_new
                in_new = np.zeros(_BASE, dtype=bool)
                in_old = np.zeros(_BASE, dtype=bool)
                in_new[pos_new] = True
                in_old[pos_old] = True
                
                only_new = in_new & ~in_old
                only_old = in_old & ~in_new
                
                gain = int(np.sum(cov[only_new] == 0))
                loss = int(np.sum(cov[only_old] == 1))
                
                delta = gain - loss
                if delta > best_delta:
                    best_delta = delta
                    best_r2 = r2
            
            if best_r2 != r and best_delta > 0:
                pos_old = _get_positions(m, r)
                pos_new = _get_positions(m, best_r2)
                
                in_old = np.zeros(_BASE, dtype=bool)
                in_new = np.zeros(_BASE, dtype=bool)
                in_old[pos_old] = True
                in_new[pos_new] = True
                
                only_old = in_old & ~in_new
                only_new = in_new & ~in_old
                
                cov[only_old] -= 1
                cov[only_new] += 1
                
                classes[i] = [m, best_r2]
                uncovered -= best_delta
                improved = True
                
                if time.time() > max_time:
                    break
    
    return classes, uncovered

def _random_restart_build(rng):
    """Build with random ordering of moduli."""
    order = _DIVS_SORTED[:]
    rng.shuffle(order)
    return _greedy_build(order)

def _hill_climb(classes, max_time):
    """More aggressive local search with multiple passes."""
    best_classes, best_uncovered = _local_refine(classes, max_time)
    return best_classes, best_uncovered

def construct():
    global _START
    
    # Start with greedy build (ascending moduli - smaller first covers more)
    classes, uncovered = _greedy_build()
    best_classes = [c[:] for c in classes]
    best_uncovered = uncovered
    
    # Local refinement
    if time.time() - _START < _DEADLINE - 15:
        refined_classes, refined_uncovered = _local_refine(best_classes)
        if refined_uncovered < best_uncovered:
            best_classes = [c[:] for c in refined_classes]
            best_uncovered = refined_uncovered
    
    # Try multiple random restarts with refinement
    rng = random.Random(42)
    while time.time() - _START < _DEADLINE - 5:
        rc, ru = _random_restart_build(rng)
        if ru < best_uncovered:
            # Refine the new best
            if time.time() - _START < _DEADLINE - 10:
                rc, ru = _local_refine(rc)
            if ru < best_uncovered:
                best_classes = [c[:] for c in rc]
                best_uncovered = ru
        
        # Also try refining current best periodically
        if time.time() - _START < _DEADLINE - 10:
            refined, ru = _local_refine(best_classes)
            if ru < best_uncovered:
                best_classes = [c[:] for c in refined]
                best_uncovered = ru
    
    return best_classes
