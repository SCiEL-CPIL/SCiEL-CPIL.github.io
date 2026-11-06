"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

_DEADLINE = time.time() + 280  # Safety margin before 330s kill

def _dist2(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add(point, chosen):
    """Check if adding point to chosen set creates any isosceles triangle."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _is_isosceles(point, chosen[i], chosen[j]):
                return False
    return True

def _is_maximal(chosen, all_points):
    """Check if the set is maximal - no point can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p not in chosen_set and _can_add(p, chosen):
            return False
    return True

def _greedy_random(all_points, order, chosen):
    """Continue greedy construction with given order and initial chosen set."""
    chosen_set = set(chosen)
    for p in order:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def _try_random_order(all_points, rng):
    """Try a random ordering and return the resulting maximal set size."""
    order = all_points[:]
    rng.shuffle(order)
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    # Check maximality
    if not _is_maximal(chosen, all_points):
        # Try to fix by adding points (shouldn't happen if greedy is correct)
        for p in all_points:
            if p not in chosen and _can_add(p, chosen):
                chosen.append(p)
    return len(chosen), chosen

def construct():
    start_time = time.time()
    all_points = list(product(range(N), repeat=2))
    rng = random.Random(42)
    
    best_size = len(all_points)  # Upper bound
    best_set = None
    
    # First, get a baseline with row-major greedy
    chosen_baseline = []
    for p in all_points:
        if _can_add(p, chosen_baseline):
            chosen_baseline.append(p)
    if not _is_maximal(chosen_baseline, all_points):
        for p in all_points:
            if p not in chosen_baseline and _can_add(p, chosen_baseline):
                chosen_baseline.append(p)
    best_size = len(chosen_baseline)
    best_set = chosen_baseline[:]
    
    # Try multiple random orderings
    attempts = 0
    while time.time() - start_time < _DEADLINE - start_time - 10:  # 10s safety margin
        if time.time() > _DEADLINE:
            break
        size, chosen = _try_random_order(all_points, rng)
        attempts += 1
        if size < best_size:
            best_size = size
            best_set = chosen[:]
            # Try local improvement: remove points if still maximal
            improved = True
            while improved:
                improved = False
                for i in range(len(best_set)):
                    candidate = best_set[:i] + best_set[i+1:]
                    if _is_maximal(candidate, all_points):
                        best_set = candidate
                        best_size = len(best_set)
                        improved = True
                        break
    
    return best_set
# EVOLVE-BLOCK-END
