"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from functools import lru_cache
import random
from math import gcd
from collections import defaultdict

def _compute_scores(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A = list(A)
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def _find_exponent_sets(base, n=20, max_val=10000):
    """Find all valid exponent sets of size n for given base."""
    # Find max exponent such that base^e <= max_val
    max_e = 0
    val = 1
    while val * base <= max_val:
        val *= base
        max_e += 1
    
    if max_e + 1 < n:
        return []
    
    # We need to choose n exponents from [0, max_e]
    # To minimize both sumset and product set sizes, we want exponents that are
    # well-distributed but also have structure
    
    # Strategy: try consecutive exponents first, then other patterns
    results = []
    
    # Consecutive exponents
    for start in range(0, max_e - n + 2):
        exponents = list(range(start, start + n))
        results.append(exponents)
    
    # Arithmetic progressions with different steps
    for step in range(1, 5):
        for start in range(0, max_e + 1):
            exponents = []
            e = start
            while e <= max_e and len(exponents) < n:
                exponents.append(e)
                e += step
            if len(exponents) == n:
                if exponents not in results:
                    results.append(exponents)
    
    return results

def _try_coprime_multipliers(exponents, base, n=20, max_val=10000):
    """Try small coprime multipliers for given exponents and base."""
    # Generate base powers
    powers = [base ** e for e in exponents]
    
    # Small coprime integers to try
    # We want multipliers that are coprime to base to maintain structure
    if base % 2 == 0:
        # base is even, use odd multipliers
        candidates = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
    elif base % 3 == 0:
        # base divisible by 3, use multipliers not divisible by 3
        candidates = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37]
    else:
        # Use small multipliers coprime to base
        candidates = []
        for k in range(1, 50):
            if gcd(k, base) == 1:
                candidates.append(k)
        candidates = candidates[:25]
    
    best_score = float('inf')
    best_set = None
    
    # Try different combinations of multipliers
    # First, try all multipliers = 1 (pure geometric)
    A = powers[:]
    if all(a <= max_val for a in A) and len(set(A)) == n:
        score = _compute_scores(A)
        if score < best_score:
            best_score = score
            best_set = A[:]
    
    # Try assigning multipliers to minimize collisions
    # Use a greedy approach: for each position, try multipliers that don't create collisions
    
    # Randomized search for better multiplier assignments
    random.seed(42)
    for trial in range(5000):
        multipliers = [random.choice(candidates) for _ in range(n)]
        A = [powers[i] * multipliers[i] for i in range(n)]
        
        # Check validity
        if any(a > max_val or a <= 0 for a in A):
            continue
        if len(set(A)) != n:
            continue
        
        A.sort()
        score = _compute_scores(A)
        if score < best_score:
            best_score = score
            best_set = A[:]
    
    return best_set, best_score

def _diophantine_search(base, n=20, max_val=10000):
    """Search for optimal set using Diophantine structure for given base."""
    exponent_sets = _find_exponent_sets(base, n, max_val)
    
    best_score = float('inf')
    best_set = None
    
    for exponents in exponent_sets:
        set_candidate, score = _try_coprime_multipliers(exponents, base, n, max_val)
        if set_candidate is not None and score < best_score:
            best_score = score
            best_set = set_candidate
    
    return best_set, best_score

def _local_search(start_set, max_iter=30000):
    """Perform local search to improve a given set."""
    A = list(start_set)
    best_score = _compute_scores(A)
    best_A = A[:]
    
    for _ in range(max_iter):
        # Try to replace one element with another value
        i = random.randint(0, len(A) - 1)
        old_val = A[i]
        
        # Try a random new value
        new_val = random.randint(1, 10000)
        if new_val in A:
            continue
        
        A[i] = new_val
        A.sort()
        
        score = _compute_scores(A)
        if score < best_score:
            best_score = score
            best_A = A[:]
        else:
            A[i] = old_val
            A.sort()
    
    return best_A, best_score

def construct():
    # Search over different bases using Diophantine structure
    best_overall_score = float('inf')
    best_overall_set = None
    
    for base in range(2, 11):
        set_candidate, score = _diophantine_search(base)
        if set_candidate is not None and score < best_overall_score:
            best_overall_score = score
            best_overall_set = set_candidate
    
    # If no good set found, fall back to simple range
    if best_overall_set is None:
        best_overall_set = list(range(1, 21))
        best_overall_score = _compute_scores(best_overall_set)
    
    # Local search to refine
    improved_set, improved_score = _local_search(best_overall_set, max_iter=20000)
    
    if improved_score < best_overall_score:
        return improved_set
    else:
        return best_overall_set

# EVOLVE-BLOCK-END
