"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

DEADLINE = 325  # seconds, safety margin before 330s kill

def _d2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy

def _can_add(p, chosen):
    """Check if p can be added to chosen without creating an isosceles triangle."""
    if not chosen:
        return True
    n = len(chosen)
    for i in range(n):
        d_pi = _d2(p, chosen[i])
        for j in range(i + 1, n):
            d_pj = _d2(p, chosen[j])
            d_ij = _d2(chosen[i], chosen[j])
            if d_pi == d_pj or d_pi == d_ij or d_pj == d_ij:
                return False
    return True

def _is_maximal(chosen, all_points):
    """Check if no point outside chosen can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True

def _greedy_random(all_points, rng):
    """Randomized greedy: pick points in random order, add if possible."""
    order = list(all_points)
    rng.shuffle(order)
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen

def _try_remove(chosen, all_points, rng, max_iters=200):
    """Try to remove points while maintaining maximality."""
    chosen = list(chosen)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        indices = list(range(len(chosen)))
        rng.shuffle(indices)
        for idx in indices:
            candidate = chosen[:idx] + chosen[idx+1:]
            if _is_maximal(candidate, all_points):
                chosen = candidate
                improved = True
                break
    return chosen

def _branch_and_bound(all_points, upper_bound, rng, time_limit):
    """
    Branch and bound to find minimum maximal no-isosceles set.
    Uses iterative deepening with random restarts and pruning.
    """
    start_time = time.time()
    best = None
    best_size = upper_bound
    
    # Get an initial solution
    initial = _greedy_random(all_points, rng)
    best = list(initial)
    best_size = len(best)
    
    current = list(best)
    
    # Try to improve by removing points and repairing
    while time.time() - start_time < time_limit:
        if not current:
            break
            
        # Remove 1-3 random points
        n_remove = rng.randint(1, min(3, len(current)))
        indices = rng.sample(range(len(current)), n_remove)
        indices.sort(reverse=True)
        for idx in indices:
            del current[idx]
        
        # Try to restore maximality by adding points
        current_set = set(current)
        remaining = [p for p in all_points if p not in current_set]
        rng.shuffle(remaining)
        
        for p in remaining:
            if len(current) >= best_size:
                break
            if _can_add(p, current):
                current.append(p)
        
        # Check maximality
        if _is_maximal(current, all_points) and len(current) < best_size:
            best = list(current)
            best_size = len(current)
    
    return best

def construct():
    start_time = time.time()
    all_points = list(itertools.product(range(N), repeat=2))
    rng = random.Random(42)
    
    best = None
    best_size = N * N + 1
    
    # Phase 1: Multiple randomized greedy starts
    phase1_deadline = DEADLINE * 0.2
    while time.time() - start_time < phase1_deadline:
        candidate = _greedy_random(all_points, rng)
        if len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
    
    # Phase 2: Local search (removal) on best candidate
    if best is not None:
        phase2_deadline = DEADLINE * 0.4
        while time.time() - start_time < phase2_deadline:
            improved = _try_remove(best, all_points, rng, max_iters=100)
            if len(improved) < best_size:
                best = improved
                best_size = len(best)
            else:
                break
        
        # Phase 3: Branch and bound
        phase3_deadline = DEADLINE * 0.9
        if best is not None:
            result = _branch_and_bound(all_points, best_size, rng, phase3_deadline - time.time())
            if result is not None and len(result) < best_size:
                best = result
                best_size = len(best)
    
    # Final: ensure maximality
    if best is not None and time.time() - start_time < DEADLINE - 5:
        if not _is_maximal(best, all_points):
            remaining = [p for p in all_points if p not in set(best)]
            rng.shuffle(remaining)
            for p in remaining:
                if _can_add(p, best):
                    best.append(p)
                if _is_maximal(best, all_points):
                    break
    
    return best if best is not None else []
# EVOLVE-BLOCK-END
