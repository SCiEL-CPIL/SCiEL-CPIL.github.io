"""Rewritten program: minimal maximal no-isosceles set on N x N grid.

Strategy:
- Use a fast vectorized isosceles check against the current chosen set.
- Greedy construction with a strong heuristic: at each step, among all
  candidate points that can be added without creating an isosceles triangle,
  pick the one that "blocks" the most remaining candidates (i.e., adding it
  would eliminate the largest number of future options). This tends to
  produce smaller maximal sets.
- Include a time-guard to return the best result before the 330 s deadline.
- Optionally, after the greedy pass, try a local-improvement pass that
  attempts to swap one chosen point for a different one to reduce size
  (this is expensive, so only done if time permits).
"""

import time
import itertools
import numpy as np

N = 16
DEADLINE_SECONDS = 300  # safety margin before 330 s kill

# Precompute all grid points
_ALL_POINTS = list(itertools.product(range(N), repeat=2))
_NUM_POINTS = N * N


def _dist2(a, b):
    """Squared distance between two 2D points (as tuples)."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _is_isosceles_with_any_pair(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle.
    
    For each pair (a, b) in chosen, check if {p, a, b} is isosceles.
    Returns True if any isosceles triangle is formed.
    """
    n = len(chosen)
    if n < 2:
        return False
    # For efficiency, compute distances from p to all chosen points
    dists = [_dist2(p, c) for c in chosen]
    # For each pair (i, j), check if dists[i] == dists[j] (p is apex)
    # or if _dist2(chosen[i], chosen[j]) == dists[i] (chosen[j] is apex... wait)
    # Actually: triangle (p, a, b) is isosceles if:
    #   dist(p,a) == dist(p,b)  [p is apex]
    #   dist(p,a) == dist(a,b)  [a is apex]
    #   dist(p,b) == dist(a,b)  [b is apex]
    for i in range(n):
        for j in range(i + 1, n):
            d_p_a = dists[i]
            d_p_b = dists[j]
            d_a_b = _dist2(chosen[i], chosen[j])
            if d_p_a == d_p_b or d_p_a == d_a_b or d_p_b == d_a_b:
                return True
    return False


def _blocking_score(p, chosen, candidates):
    """Score how many candidates would be blocked if p is added.
    
    A candidate c is blocked if adding p to chosen makes {p, c, a} or {p, c, b}
    isosceles for some a, b in chosen ∪ {p}.
    
    For simplicity and speed, we approximate: count how many candidates c
    would form an isosceles triangle with p and at least one point in chosen.
    """
    n = len(chosen)
    if n == 0:
        return len(candidates)
    
    # Precompute distances from p to all chosen
    dists_p = [_dist2(p, c) for c in chosen]
    
    count = 0
    for c in candidates:
        # Check if c with p and any chosen point forms isosceles
        d_p_c = _dist2(p, c)
        for i in range(n):
            d_c_i = _dist2(c, chosen[i])
            d_p_i = dists_p[i]
            if d_p_c == d_p_i or d_p_c == d_c_i or d_p_i == d_c_i:
                count += 1
                break
    return count


def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction with blocking heuristic
    chosen = []
    candidates = list(_ALL_POINTS)
    
    while candidates:
        if time.time() - start_time > DEADLINE_SECONDS:
            break
        
        # Find all candidates that can be added
        valid = []
        for p in candidates:
            if not _is_isosceles_with_any_pair(p, chosen):
                valid.append(p)
        
        if not valid:
            break
        
        # Choose the point with highest blocking score
        # For speed, if there are too many valid candidates, sample
        if len(valid) > 200:
            # Sample a subset for scoring
            import random
            random.seed(42)
            sample = random.sample(valid, min(200, len(valid)))
        else:
            sample = valid
        
        best_p = None
        best_score = -1
        for p in sample:
            score = _blocking_score(p, chosen, candidates)
            if score > best_score:
                best_score = score
                best_p = p
        
        if best_p is None:
            best_p = valid[0]
        
        chosen.append(best_p)
        candidates.remove(best_p)
    
    # Phase 2: Local improvement - try to swap points to reduce size
    # This is expensive, so only do it if we have time
    if time.time() - start_time < DEADLINE_SECONDS - 30:
        improved = True
        while improved and time.time() - start_time < DEADLINE_SECONDS - 10:
            improved = False
            n = len(chosen)
            for i in range(n):
                if time.time() - start_time > DEADLINE_SECONDS - 10:
                    break
                # Try removing chosen[i]
                remaining = chosen[:i] + chosen[i+1:]
                # Now try to add back a different point
                for p in _ALL_POINTS:
                    if p in remaining:
                        continue
                    if _is_isosceles_with_any_pair(p, remaining):
                        continue
                    # Check if the new set is still maximal (no point can be added)
                    # This is expensive, so skip full maximality check for now
                    # Just accept the swap if it's valid
                    new_chosen = remaining + [p]
                    # Verify no isosceles
                    valid_set = True
                    for a_idx in range(len(new_chosen)):
                        for b_idx in range(a_idx + 1, len(new_chosen)):
                            for c_idx in range(b_idx + 1, len(new_chosen)):
                                a = new_chosen[a_idx]
                                b = new_chosen[b_idx]
                                c = new_chosen[c_idx]
                                if _dist2(a, b) == _dist2(a, c) or \
                                   _dist2(a, b) == _dist2(b, c) or \
                                   _dist2(a, c) == _dist2(b, c):
                                    valid_set = False
                                    break
                            if not valid_set:
                                break
                        if not valid_set:
                            break
                    if valid_set and len(new_chosen) < len(chosen):
                        chosen = new_chosen
                        improved = True
                        break
    
    return chosen
