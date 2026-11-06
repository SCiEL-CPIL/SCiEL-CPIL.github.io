"""Minimum maximal no-isosceles set: randomized greedy + local search (removal)."""
import time
import itertools
import numpy as np

N = 16
DEADLINE = time.time() + 320.0  # safety margin before 330s kill


def _can_add(pt, chosen):
    """Return True if adding pt to chosen keeps the set no-isosceles."""
    if len(chosen) < 2:
        return True
    n = len(chosen)
    for i in range(n):
        for j in range(i + 1, n):
            a = chosen[i]
            b = chosen[j]
            d1 = (pt[0] - a[0]) ** 2 + (pt[1] - a[1]) ** 2
            d2 = (pt[0] - b[0]) ** 2 + (pt[1] - b[1]) ** 2
            d3 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if d1 == d2 or d1 == d3 or d2 == d3:
                return False
    return True


def _is_maximal(chosen):
    """Return True if chosen is maximal, else False."""
    chosen_set = set(chosen)
    n = len(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p in chosen_set:
            continue
        # Check if p can be added: no isosceles with any pair in chosen
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                a = chosen[i]
                b = chosen[j]
                d1 = (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2
                d2 = (p[0] - b[0]) ** 2 + (p[1] - b[1]) ** 2
                d3 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                if d1 == d2 or d1 == d3 or d2 == d3:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return False
    return True


def _removal_local_search(chosen):
    """Try to remove points that preserve maximality. Return new list."""
    chosen = list(chosen)
    improved = True
    while improved:
        improved = False
        for i in range(len(chosen)):
            # Time check
            if time.time() > DEADLINE - 10:
                return chosen
            candidate = chosen[:i] + chosen[i + 1:]
            if _is_maximal(candidate):
                chosen = candidate
                improved = True
                break
    return chosen


def _greedy_random():
    """Build a maximal no-isosceles set via randomized greedy."""
    points = list(itertools.product(range(N), repeat=2))
    np.random.shuffle(points)
    chosen = []
    for p in points:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def construct():
    best = None
    best_len = float('inf')
    attempts = 0
    while time.time() < DEADLINE:
        attempts += 1
        # Randomized greedy construction
        chosen_greedy = _greedy_random()
        
        # Local search: try removals
        # Save a copy in case local search fails or produces invalid result
        chosen_ls = _removal_local_search(chosen_greedy)
        
        # Verify maximality of the local search result
        if _is_maximal(chosen_ls):
            current_best = chosen_ls
        else:
            # Fallback to greedy which is guaranteed maximal
            current_best = chosen_greedy
            
        if len(current_best) < best_len:
            best_len = len(current_best)
            best = list(current_best)
            
        # Early exit if we find a very good solution
        if best_len <= 20:
            break
            
    if best is None:
        # Fallback: deterministic greedy
        chosen = []
        for p in itertools.product(range(N), repeat=2):
            if _can_add(p, chosen):
                chosen.append(p)
        return chosen
    return best
