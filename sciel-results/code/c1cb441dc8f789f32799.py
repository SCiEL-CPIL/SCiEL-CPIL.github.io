"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

_BASE = 51975
_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _divisors():
    ds = []
    for d in range(3, _BASE + 1, 2):
        if _BASE % d == 0:
            ds.append(d)
    return ds

def _score(covered):
    # number of uncovered residues
    return int(np.count_nonzero(covered == 0))

def _compute_gains(covered, divisors):
    """For each divisor, find the best residue and its gain."""
    gains = {}
    for m in divisors:
        best_r, best_gain = 0, -1
        for r in range(m):
            g = int(np.count_nonzero(covered[r::m] == 0))
            if g > best_gain:
                best_gain = g
                best_r = r
        gains[m] = (best_gain, best_r)
    return gains

def _greedy_phase(covered, remaining, divisors):
    """Greedy: repeatedly pick (m,r) maximizing new coverage."""
    classes = []
    gain_cache = _compute_gains(covered, list(remaining))
    
    while remaining:
        if time.time() > _DEADLINE - 5:
            break
        # pick m with max gain
        best_m, best = None, None
        for m in list(remaining):
            g, r = gain_cache[m]
            if g <= 0:
                continue
            if best is None or g > best[0] or (g == best[0] and m < best[1]):
                best = (g, m, r)
        if best is None:
            break
        g, m, r = best
        covered[r::m] = 1
        classes.append([m, r])
        remaining.discard(m)
        # Recompute gains for remaining divisors
        gain_cache = _compute_gains(covered, list(remaining))
    
    return classes

def _local_search(covered, classes, remaining, divisors):
    """Local search: try to improve by swapping residues or adding/removing classes."""
    best_score = _score(covered)
    best_classes = [c[:] for c in classes]
    best_remaining = set(remaining)
    
    # Try swapping residues for existing classes
    for i in range(len(classes)):
        m = classes[i][0]
        old_r = classes[i][1]
        # Try all other residues for this modulus
        for r in range(m):
            if r == old_r:
                continue
            # Temporarily remove old class
            covered[old_r::m] = 0
            # Add new class
            covered[r::m] = 1
            new_score = _score(covered)
            if new_score < best_score:
                best_score = new_score
                best_classes = [c[:] for c in classes]
                best_classes[i] = [m, r]
                best_remaining = set(remaining)
                # Keep the new state
                classes[i] = [m, r]
                # Continue trying more improvements
                break
            else:
                # Revert
                covered[r::m] = 0
                covered[old_r::m] = 1
    
    # Try adding classes from remaining
    gain_cache = _compute_gains(covered, list(remaining))
    while remaining:
        if time.time() > _DEADLINE - 5:
            break
        # pick m with max gain
        best_m, best = None, None
        for m in list(remaining):
            g, r = gain_cache[m]
            if g <= 0:
                continue
            if best is None or g > best[0] or (g == best[0] and m < best[1]):
                best = (g, m, r)
        if best is None:
            break
        g, m, r = best
        covered[r::m] = 1
        classes.append([m, r])
        remaining.discard(m)
        best_score = _score(covered)
        best_classes = [c[:] for c in classes]
        best_remaining = set(remaining)
        gain_cache = _compute_gains(covered, list(remaining))
    
    return best_classes, best_remaining, best_score

def construct():
    divisors = _divisors()
    covered = np.zeros(_BASE, dtype=np.uint8)
    remaining = set(divisors)
    
    # Phase 1: Greedy
    classes = _greedy_phase(covered, remaining, divisors)
    
    # Phase 2: Local search
    if time.time() < _DEADLINE - 5:
        classes, remaining, score = _local_search(covered, classes, remaining, divisors)
    
    # Ensure we return at least one class if possible
    if not classes and divisors:
        m = min(divisors)
        r = 0
        classes = [[m, r]]
    
    return classes
# EVOLVE-BLOCK-END
