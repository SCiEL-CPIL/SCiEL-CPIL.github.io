"""Minimum maximal no-isosceles set construction."""
N = 16
import random
import time
import itertools
from collections import defaultdict


def _dist_sq(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy


def _creates_isosceles(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle."""
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        d_pa = _dist_sq(p, a)
        for j in range(i + 1, n):
            b = chosen[j]
            d_pb = _dist_sq(p, b)
            d_ab = _dist_sq(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False


def _is_maximal(chosen):
    """Check if chosen is a maximal no-isosceles set."""
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p in chosen_set:
            continue
        if not _creates_isosceles(p, chosen):
            return False
    return True


def _greedy_build(order, chosen):
    """Try to add points from order to chosen, maintaining no-isosceles."""
    for p in order:
        if p not in set(chosen):
            if not _creates_isosceles(p, chosen):
                chosen.append(p)
    return chosen


def _local_search(chosen, deadline, rng):
    """Try to improve the set by removing points and rebuilding."""
    best = list(chosen)
    best_size = len(best)
    
    while time.time() < deadline:
        # Try removing one point and greedily rebuilding
        if len(best) <= 1:
            break
        
        # Pick a random point to remove
        idx = rng.randint(0, len(best) - 1)
        candidate = [p for i, p in enumerate(best) if i != idx]
        
        # Try to rebuild greedily with a random order
        remaining = [p for p in itertools.product(range(N), repeat=2) if p not in set(candidate)]
        rng.shuffle(remaining)
        
        for p in remaining:
            if not _creates_isosceles(p, candidate):
                candidate.append(p)
        
        if len(candidate) < best_size and _is_maximal(candidate):
            best = candidate
            best_size = len(best)
        
        # Also try: remove a point, then try to add a different point
        if time.time() < deadline:
            candidate2 = list(best)
            idx2 = rng.randint(0, len(candidate2) - 1)
            removed = candidate2.pop(idx2)
            
            # Try all other points to see if we can swap
            improved = False
            for p in itertools.product(range(N), repeat=2):
                if p == removed or p in candidate2:
                    continue
                test = candidate2 + [p]
                if not _creates_isosceles(p, candidate2):
                    # Check if this new set is maximal
                    if _is_maximal(test):
                        best = test
                        best_size = len(best)
                        improved = True
                        break
            
            if not improved:
                candidate2.append(removed)
    
    return best


def construct():
    deadline = time.time() + 280  # Safety margin before 330s limit
    rng = random.Random(42)
    
    best = None
    best_size = float('inf')
    
    # Multiple random restarts with greedy construction
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random order for greedy
        all_points = list(itertools.product(range(N), repeat=2))
        rng.shuffle(all_points)
        
        chosen = []
        for p in all_points:
            if not _creates_isosceles(p, chosen):
                chosen.append(p)
        
        # Check maximality
        if _is_maximal(chosen):
            if len(chosen) < best_size:
                best_size = len(chosen)
                best = list(chosen)
        
        # Local search on this solution
        if time.time() < deadline - 60:
            improved = _local_search(chosen, deadline - 30, rng)
            if len(improved) < best_size:
                best_size = len(improved)
                best = improved
    
    # If no maximal set found, do a final greedy and verify
    if best is None:
        chosen = []
        for p in itertools.product(range(N), repeat=2):
            if not _creates_isosceles(p, chosen):
                chosen.append(p)
        best = chosen
    
    return best
