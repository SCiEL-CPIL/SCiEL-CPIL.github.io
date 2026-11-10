"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
from itertools import combinations
import time

# EVOLVE-BLOCK-START

def _subset_sum_exists(target, candidates):
    """Check if target can be formed as a subset sum of candidates (exact fractions).
    Uses DP with frozenset memoization for small candidate sets."""
    if target == 0:
        return True
    if not candidates:
        return False
    
    # For small sets, use exhaustive subset enumeration
    n = len(candidates)
    if n <= 20:
        # Enumerate all subsets
        from itertools import combinations
        for r in range(n + 1):
            for combo in combinations(range(n), r):
                s = Fraction(0)
                for i in combo:
                    s += candidates[i]
                if s == target:
                    return True
        return False
    else:
        # Fallback: recursive with pruning
        candidates_sorted = sorted(candidates, reverse=True)
        
        def _rec(idx, remaining):
            if remaining == 0:
                return True
            if idx >= len(candidates_sorted):
                return False
            if remaining < 0:
                return False
            c = candidates_sorted[idx]
            if c <= remaining:
                if _rec(idx + 1, remaining - c):
                    return True
            return _rec(idx + 1, remaining)
        
        return _rec(0, target)


def _is_conflict(x, S):
    """Check if adding x to set S creates a conflict."""
    rx = Fraction(1, x)
    reciprocals_S = [Fraction(1, y) for y in S]
    
    # Check 1: is 1/x a subset sum of reciprocals of S?
    if _subset_sum_exists(rx, reciprocals_S):
        return True
    
    # Check 2: for each y in S, is 1/y a subset sum of {1/x} ∪ others?
    for i, y in enumerate(S):
        ry = Fraction(1, y)
        others = [reciprocals_S[j] for j in range(len(S)) if j != i]
        candidates = [rx] + others
        if _subset_sum_exists(ry, candidates):
            return True
    
    return False


def _greedy_descending(elements):
    """Greedy: add elements in descending order if no conflict."""
    result = []
    for x in elements:
        if not _is_conflict(x, result):
            result.append(x)
    return result


def _backtrack_search(elements, deadline, max_time_per_node=0.001):
    """Backtracking search with time limit."""
    best_set = []
    best_size = 0
    nodes_visited = 0
    
    def _search(idx, current_set, current_size):
        nonlocal best_set, best_size, nodes_visited
        
        if time.time() > deadline:
            return
        
        nodes_visited += 1
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
        
        if idx >= len(elements):
            return
        
        # Upper bound pruning
        remaining = len(elements) - idx
        if current_size + remaining <= best_size:
            return
        
        x = elements[idx]
        
        # Try including x first (prefer larger elements)
        if not _is_conflict(x, current_set):
            current_set.append(x)
            _search(idx + 1, current_set, current_size + 1)
            current_set.pop()
        
        # Try excluding x
        _search(idx + 1, current_set, current_size)
    
    _search(0, [], 0)
    return best_set


def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin
    
    # Elements in descending order
    elements = list(range(24, 0, -1))
    
    # First, get a good initial solution via greedy
    greedy_result = _greedy_descending(elements)
    best_set = greedy_result
    best_size = len(greedy_result)
    
    # Try to improve with backtracking
    if time.time() < deadline - 5.0:  # Only if we have time
        result = _backtrack_search(elements, deadline)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
    
    # Safety: ensure nonempty and valid
    if not best_set:
        best_set = [24]
    
    return list(best_set)

# EVOLVE-BLOCK-END
