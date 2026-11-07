"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

# Primes up to 40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NUM_PRIMES = len(PRIMES)

def get_vector(n):
    v = [0] * NUM_PRIMES
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            v[i] += 1
            n //= p
    return tuple(v)

# Precompute vectors for all candidates 2..40
CANDIDATES = list(range(2, 41))
VECTORS = {n: get_vector(n) for n in CANDIDATES}

# We want to find the largest subset of CANDIDATES such that all subset sums of their vectors are distinct.
# This is equivalent to: for any distinct subsets A, B, sum(A) != sum(B).
# Which is equivalent to: sum(A \ B) != sum(B \ A) for disjoint sets.
# Or simply: 0 is not in the set of differences of distinct subset sums.

best_set = []

def add_vector(S, v):
    # S is a set of tuples (subset sums)
    # Returns new set S' = S U {s+v for s in S} if valid, else None
    # Check if any s+v is already in S
    # Optimization: if we sort S, we can do two pointers? 
    # But S is a set of tuples, not sorted.
    # Converting to sorted list every time is expensive.
    # Let's just iterate.
    new_elements = []
    for s in S:
        ns = tuple(s[i] + v[i] for i in range(NUM_PRIMES))
        if ns in S:
            return None
        new_elements.append(ns)
    return S.union(new_elements)

def search(idx, current_S, current_set):
    global best_set
    
    # Pruning: if even taking all remaining candidates doesn't beat best, return
    remaining = len(CANDIDATES) - idx
    if len(current_set) + remaining <= len(best_set):
        return

    if idx == len(CANDIDATES):
        if len(current_set) > len(best_set):
            best_set = current_set[:]
        return

    # Option 1: Skip current candidate
    search(idx + 1, current_S, current_set)

    # Option 2: Include current candidate
    v = VECTORS[CANDIDATES[idx]]
    new_S = add_vector(current_S, v)
    if new_S is not None:
        current_set.append(CANDIDATES[idx])
        search(idx + 1, new_S, current_set)
        current_set.pop()

def construct():
    global best_set
    best_set = []
    
    # Initialize with empty set of sums {0}
    initial_S = {tuple([0]*NUM_PRIMES)}
    
    # To improve search efficiency, we can sort candidates.
    # Heuristic: sort by the "magnitude" of the vector? 
    # Or just numerical order.
    # Let's try numerical order first.
    # If we pick small numbers first, the subset sums might stay small?
    # Actually, the vectors are sparse.
    
    # A better heuristic for branch and bound for MAX size:
    # Try including first? Or skipping first?
    # Usually, to find a large set, we should try to include as many as possible.
    # So we should try "Include" first in the recursion?
    # If we include, we might hit a conflict earlier and prune?
    # Or we might build a valid large set quickly.
    # Let's try "Include" first.
    
    def search_optimized(idx, current_S, current_set):
        global best_set
        
        remaining = len(CANDIDATES) - idx
        if len(current_set) + remaining <= len(best_set):
            return

        if idx == len(CANDIDATES):
            if len(current_set) > len(best_set):
                best_set = current_set[:]
            return

        # Try Include first
        v = VECTORS[CANDIDATES[idx]]
        new_S = add_vector(current_S, v)
        if new_S is not None:
            current_set.append(CANDIDATES[idx])
            search_optimized(idx + 1, new_S, current_set)
            current_set.pop()
            
        # Try Skip
        search_optimized(idx + 1, current_S, current_set)

    # Sort candidates? 
    # If we sort by vector "complexity" (number of primes), maybe?
    # Primes have 1 prime factor. Composites have more.
    # Vectors with fewer non-zero components might be "simpler" to pack?
    # Or maybe the opposite.
    # Let's stick to numerical order for now, or sort by sum of exponents.
    
    # Let's sort by the number of prime factors (with multiplicity)
    def key_func(n):
        v = VECTORS[n]
        return sum(v)
        
    CANDIDATES_SORTED = sorted(CANDIDATES, key=key_func)
    # Update VECTORS reference? No, VECTORS is a dict.
    # We just iterate over CANDIDATES_SORTED.
    
    # We need to adapt search to use the sorted list.
    def search_sorted(idx, current_S, current_set):
        global best_set
        
        remaining = len(CANDIDATES_SORTED) - idx
        if len(current_set) + remaining <= len(best_set):
            return

        if idx == len(CANDIDATES_SORTED):
            if len(current_set) > len(best_set):
                best_set = current_set[:]
            return

        # Try Include first
        n = CANDIDATES_SORTED[idx]
        v = VECTORS[n]
        new_S = add_vector(current_S, v)
        if new_S is not None:
            current_set.append(n)
            search_sorted(idx + 1, new_S, current_set)
            current_set.pop()
            
        # Try Skip
        search_sorted(idx + 1, current_S, current_set)

    initial_S = {tuple([0]*NUM_PRIMES)}
    search_sorted(0, initial_S, [])
    
    # Ensure result is sorted list of ints
    return [int(x) for x in sorted(best_set)]

# EVOLVE-BLOCK-END
