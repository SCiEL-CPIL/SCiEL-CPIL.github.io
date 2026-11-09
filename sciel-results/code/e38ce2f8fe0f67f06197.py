"""Improved program for minimum maximal no-isosceles set."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280  # safety margin before 330s limit


def _d2(a, b):
    """Squared distance between two points."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_isosceles(a, b, c):
    """Check if triangle (a,b,c) is isosceles."""
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(p, chosen):
    """Check if point p can be added to chosen set without creating isosceles triangles."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _is_isosceles(p, chosen[i], chosen[j]):
                return False
    return True


def _is_maximal(chosen, all_points):
    """Check if no further point can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p not in chosen_set:
            if _can_add(p, chosen):
                return False
    return True


def _greedy_construct(order):
    """Greedy construction following a given point order."""
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    # Make it maximal: add any remaining points that can be added
    chosen_set = set(chosen)
    for p in order:
        if p not in chosen_set:
            if _can_add(p, chosen):
                chosen.append(p)
                chosen_set.add(p)
    return chosen


def _local_search(chosen, max_iterations=500):
    """Local search: try to remove points and re-add different ones to reduce size."""
    chosen = list(chosen)
    best = list(chosen)
    
    for iteration in range(max_iterations):
        if time.time() > _DEADLINE:
            break
        
        improved = False
        # Try removing each point and see if we can get a smaller maximal set
        for i in range(len(chosen)):
            if time.time() > _DEADLINE:
                break
            
            # Try removing point i
            candidate = chosen[:i] + chosen[i+1:]
            if not candidate:
                continue
            
            # Check if candidate is still no-isosceles (it should be, since it's a subset)
            # Now try to make it maximal by adding points
            candidate_set = set(candidate)
            new_candidate = list(candidate)
            
            for p in range(N * N):
                point = (p // N, p % N)
                if point not in candidate_set:
                    if _can_add(point, new_candidate):
                        new_candidate.append(point)
                        candidate_set.add(point)
            
            if len(new_candidate) < len(best):
                best = new_candidate
                chosen = new_candidate
                improved = True
                break
        
        if not improved:
            break
    
    return best


def construct():
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Start with greedy in row-major order
    best = _greedy_construct(all_points)
    
    # Try different orderings to find smaller maximal sets
    random.seed(42)
    num_tries = 200
    
    for _ in range(num_tries):
        if time.time() > _DEADLINE:
            break
        
        # Random ordering
        shuffled = all_points[:]
        random.shuffle(shuffled)
        candidate = _greedy_construct(shuffled)
        
        if len(candidate) < len(best):
            best = candidate
    
    # Local search on the best found
    best = _local_search(best)
    
    return best
# EVOLVE-BLOCK-END
