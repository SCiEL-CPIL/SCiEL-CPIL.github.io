"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_BASE = 51975
_DEADLINE = 145.0  # safety margin before 150s kill

def _get_divisors(n):
    """Get all divisors of n greater than 1."""
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    divs.discard(1)
    return sorted(divs)

def _compute_uncovered(classes):
    """Compute number of uncovered residues for given classes."""
    covered = np.zeros(_BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::_BASE // m * m] = 1  # This is wrong, fix below
    # Proper way:
    covered = np.zeros(_BASE, dtype=np.uint8)
    for m, r in classes:
        idx = np.arange(r, _BASE, m)
        covered[idx] = 1
    return int(np.sum(covered == 0))

def _greedy_build():
    """Greedy construction: for each modulus, pick residue covering most uncovered."""
    covered = np.zeros(_BASE, dtype=np.uint8)
    divisors = _get_divisors(_BASE)
    classes = []
    
    for m in divisors:
        # Find residue r that covers most currently-uncovered residues
        best_r = 0
        best_count = -1
        for r in range(m):
            idx = np.arange(r, _BASE, m)
            count = int(np.sum(covered[idx] == 0))
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        idx = np.arange(best_r, _BASE, m)
        covered[idx] = 1
    
    return classes

def _local_search(classes, time_limit=10.0):
    """Local search: try swapping residues to reduce uncovered count."""
    start_time = time.time()
    current_classes = [list(c) for c in classes]
    current_score = _compute_uncovered(current_classes)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i, (m, r) in enumerate(current_classes):
            for r_new in range(m):
                if r_new == r:
                    continue
                old_r = current_classes[i][1]
                current_classes[i][1] = r_new
                new_score = _compute_uncovered(current_classes)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current_classes[i][1] = old_r
                if (time.time() - start_time) > time_limit:
                    return current_classes, current_score
    
    return current_classes, current_score

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    classes = _greedy_build()
    score = _compute_uncovered(classes)
    
    # Phase 2: Local search with time budget
    if time.time() - start_time < _DEADLINE - 5:
        classes, score = _local_search(classes, time_limit=_DEADLINE - time.time() - 3)
    
    # Final deadline check
    if time.time() - start_time > _DEADLINE:
        # Return whatever we have (shouldn't happen due to guards)
        pass
    
    return classes
# EVOLVE-BLOCK-END
