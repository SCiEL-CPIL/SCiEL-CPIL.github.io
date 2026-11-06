import time
from math import gcd
from functools import reduce
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    return reduce(_lcm, lst, 1)

def _is_subsum_free(selected):
    """Check if no element's reciprocal equals sum of reciprocals of a subset of others."""
    if len(selected) <= 1:
        return True
    # For each element s, check if 1/s can be formed by subset of others
    # Use LCM-scaled integers to avoid floating point
    for s in selected:
        others = [x for x in selected if x != s]
        if not others:
            continue
        # Check all subsets of others (up to size len(others))
        # 1/s = sum(1/t for t in subset)
        # Multiply by LCM(s, others): LCM/s = sum(LCM/t for t in subset)
        lcm_all = _lcm_list([s] + others)
        target = lcm_all // s
        weights = [lcm_all // t for t in others]
        # Check if any subset of weights sums to target
        # Use meet-in-the-middle or simple subset sum for small sets
        n = len(weights)
        if n > 20:
            # For large sets, use a simpler check: just check small subsets
            # If target can be formed, it's likely with few elements
            for r in range(1, min(n, 8) + 1):
                for subset in combinations(range(n), r):
                    if sum(weights[i] for i in subset) == target:
                        return False
            # Also check full set and near-full
            if sum(weights) == target:
                return False
        else:
            # Brute force all subsets
            for r in range(1, n + 1):
                for subset in combinations(range(n), r):
                    if sum(weights[i] for i in subset) == target:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin
    
    # Start with all elements 1-24
    selected = list(range(1, 25))
    
    # Iteratively remove elements that cause violations
    # Priority: remove elements that are "most problematic"
    while True:
        if time.time() > deadline:
            break
        if not _is_subsum_free(selected):
            # Find which elements cause violations
            # For each s in selected, check if 1/s is subsum of others
            problematic = []
            for s in selected:
                others = [x for x in selected if x != s]
                if not others:
                    continue
                lcm_all = _lcm_list([s] + others)
                target = lcm_all // s
                weights = [lcm_all // t for t in others]
                n = len(weights)
                found = False
                if n <= 20:
                    for r in range(1, n + 1):
                        for subset in combinations(range(n), r):
                            if sum(weights[i] for i in subset) == target:
                                found = True
                                break
                        if found:
                            break
                else:
                    for r in range(1, min(n, 8) + 1):
                        for subset in combinations(range(n), r):
                            if sum(weights[i] for i in subset) == target:
                                found = True
                                break
                        if found:
                            break
                    if not found and sum(weights) == target:
                        found = True
                if found:
                    problematic.append(s)
            
            if not problematic:
                break
            
            # Remove the most problematic element (smallest index as tiebreaker)
            # Actually, remove the one that causes the most conflicts
            # Simple heuristic: remove the smallest problematic element
            selected.remove(min(problematic))
        else:
            break
    
    return selected
