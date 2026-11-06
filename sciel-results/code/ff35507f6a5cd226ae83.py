"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_valid(S):
    """Check if S is reciprocal-subsum-free."""
    S_set = set(S)
    for x in S:
        # Check if 1/x equals sum of reciprocals of any non-empty subset of S\{x}
        others = [y for y in S if y != x]
        if not others:
            continue
        target = 1.0 / x
        # Check all non-empty subsets of others
        # For efficiency, we can use a set of achievable sums
        # But for small sets, we can just iterate
        # Use a smarter approach: build set of achievable sums
        achievable = {0.0}
        for y in others:
            new_sums = set()
            for s in achievable:
                ns = s + 1.0/y
                new_sums.add(ns)
            achievable |= new_sums
        if target in achievable:
            return False
    return True

def _score(S):
    """Return cardinality if valid, else 0."""
    if not S:
        return 0
    if _is_valid(S):
        return len(S)
    return 0

def construct():
    deadline = time.time() + 80.0  # 80s deadline, 10s margin
    best_set = list(range(13, 25))  # Fallback: size 12
    best_score = 12
    
    # Greedy: start with all elements, try to add back removed ones
    # Or: start empty, add largest first
    S = []
    # Add from largest to smallest
    for x in range(24, 0, -1):
        if time.time() > deadline - 10:
            break
        candidate = S + [x]
        if _is_valid(candidate):
            S = candidate
    
    if len(S) > best_score:
        best_score = len(S)
        best_set = list(S)
    
    # Local search: try to improve by swapping or adding
    # First, try to add any element not in S
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for x in range(24, 0, -1):
            if x not in S:
                candidate = S + [x]
                if _is_valid(candidate):
                    S = candidate
                    best_score = len(S)
                    best_set = list(S)
                    changed = True
                    break
    
    # Try local search: remove one, add two
    if time.time() < deadline - 5:
        S_copy = list(S)
        for i in range(len(S_copy)):
            if time.time() > deadline - 5:
                break
            for j in range(24, 0, -1):
                if j in S_copy:
                    continue
                for k in range(24, 0, -1):
                    if k in S_copy or k == j:
                        continue
                    candidate = [x for x in S_copy if x != S_copy[i]] + [j, k]
                    if len(candidate) > len(S_copy) and _is_valid(candidate):
                        S = candidate
                        best_score = len(S)
                        best_set = list(S)
                        break
                if time.time() > deadline - 5:
                    break
    
    return best_set
# EVOLVE-BLOCK-END
