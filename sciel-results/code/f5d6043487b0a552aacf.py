"""Search-based covering system optimizer with local search."""
import time
import random
import math
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
_BASE = 51975
_DEADLINE = 115.0
_START = time.time()
_rng = random.Random(20240517)

def _divisors(n):
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

_DIVS = _divisors(_BASE)  # 41 distinct odd divisors > 1
_N_DIVS = len(_DIVS)

def _build_coverage(classes):
    """Build coverage mask using numpy for speed."""
    covered = np.zeros(_BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::_BASE//m] = 1  # This won't work correctly, use explicit indexing
    # Actually, let's do it properly with numpy slicing
    covered = np.zeros(_BASE, dtype=np.uint8)
    for m, r in classes:
        indices = np.arange(r, _BASE, m)
        covered[indices] = 1
    return covered

def _score(classes):
    """Exact uncovered count using numpy."""
    covered = _build_coverage(classes)
    return _BASE - int(covered.sum())

def _build_greedy(order='ascending'):
    """Greedy construction with specified order."""
    divs = sorted(_DIVS) if order == 'ascending' else sorted(_DIVS, reverse=True)
    covered = np.zeros(_BASE, dtype=np.uint8)
    classes = []
    for m in divs:
        best_r = 0
        best_count = -1
        for r in range(m):
            indices = np.arange(r, _BASE, m)
            count = int((covered[indices] == 0).sum())
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        indices = np.arange(best_r, _BASE, m)
        covered[indices] = 1
    return classes

def _build_random(seed_offset=0):
    """Random initial solution with fixed seed offset."""
    rng = random.Random(20240517 + seed_offset)
    classes = []
    for m in sorted(_DIVS):
        r = rng.randint(0, m - 1)
        classes.append([m, r])
    return classes

def _local_search(classes, max_iters=5000, deadline=_DEADLINE - 10):
    """Iteratively improve by changing residues of individual moduli with fast gain computation."""
    best_score = _score(classes)
    best_classes = [c[:] for c in classes]
    
    # Precompute coverage mask
    covered = _build_coverage(best_classes)
    
    # Precompute index arrays for each class for fast gain computation
    index_cache = {}
    for idx, (m, r) in enumerate(best_classes):
        index_cache[idx] = np.arange(r, _BASE, m)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        for idx in range(_N_DIVS):
            m, r = best_classes[idx]
            current_indices = index_cache[idx]
            # Try all possible residues for this modulus
            best_r = r
            best_gain = 0
            for new_r in range(m):
                if new_r == r:
                    continue
                new_indices = np.arange(new_r, _BASE, m)
                # Gain = newly covered - lost coverage
                newly_covered = int((covered[new_indices] == 0).sum())
                lost = int((covered[current_indices] == 0).sum())
                gain = newly_covered - lost
                if gain > best_gain:
                    best_gain = gain
                    best_r = new_r
            if best_gain > 0:
                # Update covered array incrementally
                best_classes[idx] = [m, best_r]
                index_cache[idx] = np.arange(best_r, _BASE, m)
                # Recompute covered from scratch (simpler and still fast enough)
                covered = _build_coverage(best_classes)
                best_score = _BASE - int(covered.sum())
                improved = True
                break
        if not improved:
            break
    return best_classes, best_score

def _simulated_annealing(classes, max_iters=3000, deadline=_DEADLINE - 15):
    """Simulated annealing to escape local optima."""
    current_classes = [c[:] for c in classes]
    current_score = _score(current_classes)
    best_classes = [c[:] for c in current_classes]
    best_score = current_score
    
    # Coverage mask
    covered = _build_coverage(current_classes)
    
    # Precompute index arrays
    index_cache = {}
    for idx, (m, r) in enumerate(current_classes):
        index_cache[idx] = np.arange(r, _BASE, m)
    
    T = 50.0
    T_min = 0.01
    cooling = 0.98
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if T <= T_min:
            break
            
        idx = _rng.randint(0, _N_DIVS - 1)
        m, r = current_classes[idx]
        new_r = _rng.randint(0, m - 1)
        if new_r == r:
            continue
            
        current_indices = index_cache[idx]
        new_indices = np.arange(new_r, _BASE, m)
        
        # Compute delta score
        newly_covered = int((covered[new_indices] == 0).sum())
        lost = int((covered[current_indices] == 0).sum())
        delta = lost - newly_covered  # positive means worse
        
        # Accept or reject
        if delta < 0 or _rng.random() < math.exp(-delta / T):
            # Update coverage incrementally
            # Remove old coverage contribution
            for i in current_indices:
                # Check if any other class covers this position
                still_covered = False
                for j, (mj, rj) in enumerate(current_classes):
                    if j == idx:
                        continue
                    if (i - rj) % mj == 0 and i >= rj:
                        still_covered = True
                        break
                if not still_covered:
                    covered[i] = 0
            # Add new coverage
            for i in new_indices:
                covered[i] = 1
            current_classes[idx] = [m, new_r]
            index_cache[idx] = new_indices
            current_score += delta
            if current_score < best_score:
                best_score = current_score
                best_classes = [c[:] for c in current_classes]
        
        T *= cooling
    
    # Final local search polish
    best_classes, best_score = _local_search(best_classes, max_iters=500, deadline=deadline)
    return best_classes, best_score

def _polish(classes, deadline=_DEADLINE - 5):
    """Final high-precision polish: multiple passes of local search."""
    best_classes = [c[:] for c in classes]
    best_score = _score(best_classes)
    
    for _ in range(3):
        if time.time() > deadline:
            break
        new_classes, new_score = _local_search(best_classes, max_iters=2000, deadline=deadline)
        if new_score < best_score:
            best_classes = new_classes
            best_score = new_score
        else:
            break
    
    return best_classes, best_score

def construct():
    # Phase 1: Greedy initial solution - try both orders
    classes_asc = _build_greedy('ascending')
    score_asc = _score(classes_asc)
    
    classes_desc = _build_greedy('descending')
    score_desc = _score(classes_desc)
    
    if score_asc <= score_desc:
        classes = classes_asc
        best_score = score_asc
    else:
        classes = classes_desc
        best_score = score_desc
    
    best_classes = [c[:] for c in classes]
    
    # Phase 2: Local search on greedy
    ls_classes, ls_score = _local_search(classes, max_iters=5000, deadline=_DEADLINE - 30)
    if ls_score < best_score:
        best_classes = ls_classes
        best_score = ls_score
    
    # Phase 3: Simulated annealing from best solution
    if time.time() < _DEADLINE - 40:
        sa_classes, sa_score = _simulated_annealing(best_classes, max_iters=5000, deadline=_DEADLINE - 20)
        if sa_score < best_score:
            best_classes = sa_classes
            best_score = sa_score
    
    # Phase 4: Try random starts with local search
    if time.time() < _DEADLINE - 25:
        for offset in range(1, 6):
            if time.time() > _DEADLINE - 20:
                break
            rand_classes = _build_random(offset)
            rc_classes, rc_score = _local_search(rand_classes, max_iters=3000, deadline=_DEADLINE - 15)
            if rc_score < best_score:
                best_classes = rc_classes
                best_score = rc_score
    
    # Phase 5: Final polish
    best_classes, best_score = _polish(best_classes, deadline=_DEADLINE - 3)
    
    return best_classes
# EVOLVE-BLOCK-END
