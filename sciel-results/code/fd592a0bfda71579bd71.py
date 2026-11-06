import time
import heapq
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds to be safe
    
    target = Fraction(5, 31)
    
    # Dijkstra-like search: state = (current_remainder, frozenset_of_used_denoms)
    # Cost = max denominator used so far
    # We want to reach remainder 0 with minimum max denominator
    
    # State: (max_denom_so_far, current_remainder, tuple_of_used_denoms)
    # Use a priority queue ordered by max_denom_so_far
    
    # To avoid state explosion, we'll use a different approach:
    # BFS/DFS with pruning, searching for solutions with small max denominator
    
    # Key insight: if we bound the maximum denominator to M, then all denominators
    # must be in [1, M]. We can binary search on M or do a structured search.
    
    # Alternative: Use the fact that for Egyptian fractions, if we fix the first k
    # denominators, the remainder determines constraints on subsequent denominators.
    
    # Let's do a systematic search:
    # For a given max_denom M, check if 5/31 can be written as sum of distinct
    # unit fractions with all denominators <= M.
    
    # But that's expensive. Instead, let's use a smarter approach:
    
    # We know from the current best that max_denom = 248 works (solution [8, 31, 248]).
    # Let's try to find better solutions by exploring different structures.
    
    # Approach: Enumerate possible sets of small denominators and check if the
    # remainder can be completed.
    
    best_max_denom = float('inf')
    best_solution = None
    
    def try_complete(remainder, used_denoms, max_terms=256):
        """Try to complete the Egyptian fraction decomposition.
        Returns list of additional denominators or None if impossible."""
        if remainder == 0:
            return []
        if len(used_denoms) >= max_terms:
            return None
        if remainder < 0:
            return None
        
        # If remainder is a unit fraction 1/k and k not in used, we're done
        if remainder.numerator == 1:
            k = remainder.denominator
            if k not in used_denoms:
                return [k]
            return None
        
        # Greedy with backtracking for small cases
        # Try denominators from ceil(1/remainder) upward
        min_d = (remainder.denominator + remainder.numerator - 1) // remainder.numerator
        
        # Try a limited range of next denominators
        for d in range(min_d, min_d + 50):
            if d in used_denoms:
                continue
            if d > 10000000:
                break
            new_rem = remainder - Fraction(1, d)
            if new_rem < 0:
                continue
            if new_rem == 0:
                return [d]
            new_used = used_denoms | {d}
            result = try_complete(new_rem, new_used, max_terms)
            if result is not None:
                return [d] + result
        return None
    
    # Systematic search over first 1-3 denominators
    # d1: smallest denominator
    # For 5/31, d1 must satisfy 1/d1 < 5/31, so d1 > 31/5 = 6.2, so d1 >= 7
    
    for d1 in range(7, 30):
        if time.time() > deadline:
            break
        
        rem1 = target - Fraction(1, d1)
        if rem1 <= 0:
            continue
        
        # Try to complete with remaining denominators > d1 (to ensure distinctness and ordering)
        used1 = {d1}
        result = try_complete(rem1, used1)
        if result is not None:
            candidate = [d1] + result
            max_d = max(candidate)
            if max_d < best_max_denom:
                best_max_denom = max_d
                best_solution = candidate
                if best_max_denom <= 50:
                    break  # Very good solution found
    
    # Also try 2-term solutions
    for d1 in range(7, 30):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, d1)
        if rem1 <= 0:
            continue
        if rem1.numerator == 1:
            d2 = rem1.denominator
            if d2 != d1:
                candidate = [d1, d2]
                max_d = max(candidate)
                if max_d < best_max_denom:
                    best_max_denom = max_d
                    best_solution = candidate
    
    # If we found a good solution, return it
    if best_solution is not None:
        return best_solution
    
    # Fallback: greedy decomposition
    r, values = Fraction(5, 31), []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
