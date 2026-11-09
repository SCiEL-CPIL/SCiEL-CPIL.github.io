"""Rewritten solver: maximize cardinality of S ⊆ [1,192] (|S|≤48) such that
no selected integer divides the sum of any 3 distinct OTHER selected integers."""

import time
import random
from itertools import combinations

# Global deadline guard (evaluator kills after 90s; we use a safety margin)
_DEADLINE = 85.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _check_validity(selected):
    """Return True if 'selected' satisfies the condition."""
    n = len(selected)
    if n < 3:
        return True
    # For each element a, check that a does NOT divide sum of any 3 distinct others
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        # If fewer than 3 others, no triple exists
        if len(others) < 3:
            continue
        # We need to check all triples of others
        # Optimization: if a > sum of three largest others, a cannot divide any sum (since sum < a and sum > 0)
        # Actually if sum < a and sum > 0, sum % a == sum != 0, so it's fine.
        # So we only need to check triples where sum >= a.
        # But checking all is simpler for correctness; let's try to prune.
        # For speed in validation, we can sort others descending and break early if sum of top 3 < a?
        # No, we need to check ALL triples. If a is small, many triples might sum to multiples of a.
        # If a is large, fewer triples sum to multiples of a.
        # Let's just do the full check but with a fast inner loop.
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _greedy_largest_first():
    """Greedy: add largest numbers first."""
    selected = []
    for x in range(192, 0, -1):
        if _time_left() < 1.0:
            break
        candidate = selected + [x]
        if _check_validity(candidate):
            selected.append(x)
    return selected

def _greedy_smallest_first():
    """Greedy: add smallest numbers first."""
    selected = []
    for x in range(1, 193):
        if _time_left() < 1.0:
            break
        candidate = selected + [x]
        if _check_validity(candidate):
            selected.append(x)
    return selected

def _local_search(initial, max_iter=500):
    """Hill-climbing: try to add elements or swap to increase size."""
    selected = list(initial)
    current_size = len(selected)
    best = selected[:]
    
    # Precompute set for O(1) membership
    sel_set = set(selected)
    
    # Try adding elements
    for _ in range(max_iter):
        if _time_left() < 2.0:
            break
        improved = False
        # Try adding any number not in selected
        for x in range(1, 193):
            if x in sel_set:
                continue
            candidate = selected + [x]
            if _check_validity(candidate):
                selected.append(x)
                sel_set.add(x)
                current_size += 1
                improved = True
                break
        if not improved:
            # Try swapping: remove one, add one (net 0 change in size, but might allow larger additions later?)
            # Or just break if no addition possible.
            # Let's try a simple swap: remove x, add y where y > x or just any y.
            # This is complex. Let's stick to additions for now.
            break
    return selected

def construct():
    global _START
    _START = time.time()
    
    # 1. Try deterministic greedy strategies
    candidates = []
    
    # Largest first
    cand1 = _greedy_largest_first()
    if _check_validity(cand1):
        candidates.append(cand1)
        
    # Smallest first
    cand2 = _greedy_smallest_first()
    if _check_validity(cand2):
        candidates.append(cand2)
        
    # 2. Randomized restarts with greedy + local search
    random.seed(42)
    
    best = []
    best_size = 0
    
    for cand in candidates:
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)
            
    # If time remains, do random restarts
    # Strategy: pick a random subset of size k, then greedily fill
    # Or: random order greedy
    
    # Let's try random order greedy
    for restart in range(50):
        if _time_left() < 5.0:
            break
            
        # Random permutation of [1..192]
        nums = list(range(1, 193))
        random.shuffle(nums)
        
        selected = []
        for x in nums:
            if _time_left() < 1.0:
                break
            candidate = selected + [x]
            if _check_validity(candidate):
                selected.append(x)
                
        if _check_validity(selected):
            if len(selected) > best_size:
                best = selected
                best_size = len(selected)
                
    # Try to improve best with local search (additions)
    if _time_left() > 3.0:
        improved = _local_search(best, max_iter=100)
        if _check_validity(improved) and len(improved) > best_size:
            best = improved
            best_size = len(improved)
            
    return best

# Quick self-test
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Valid: {_check_validity(res)}")
    print(res)
