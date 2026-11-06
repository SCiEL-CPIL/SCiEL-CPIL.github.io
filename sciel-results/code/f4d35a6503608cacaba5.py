import time
import random
import sys

# Global cache for priority sets
_PRIORITY_CACHE = {}

# Deadline guard
_START_TIME = time.time()
DEADLINE = _START_TIME + 310.0  # Leave 20s safety margin before 330s kill


def _int_to_tuple(idx, n):
    """Convert an integer index to a tuple of digits in base 3."""
    v = []
    x = idx
    for _ in range(n):
        v.append(x % 3)
        x //= 3
    return tuple(v)


def _tuple_to_int(v):
    """Convert a tuple of digits in base 3 to an integer index."""
    idx = 0
    for i, d in enumerate(reversed(v)):
        idx += d * (3 ** i)
    return idx


def _greedy_cap_set_fast(n, seed, time_limit):
    """
    Greedily build a cap set using integer indices for speed.
    Uses a forbidden set of integer indices.
    """
    start = time.time()
    total = 3 ** n
    
    # Shuffle indices
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    
    S = set()
    forbidden = set()
    
    for idx in indices:
        if time.time() - start > time_limit:
            break
        if idx in forbidden:
            continue
        
        S.add(idx)
        
        # Convert to tuple for arithmetic
        v = _int_to_tuple(idx, n)
        
        # For each existing element a in S (excluding v), compute the third point
        for a_idx in S:
            if a_idx == idx:
                continue
            a = _int_to_tuple(a_idx, n)
            
            # The line through a and v: third point w = 2v - a (mod 3)
            w = tuple((2 * v[i] - a[i]) % 3 for i in range(n))
            w_idx = _tuple_to_int(w)
            forbidden.add(w_idx)
    
    return S


def _greedy_with_priority(n, seed, time_limit):
    """
    Greedy with a static priority ordering.
    Vectors are processed in order of a priority function, not random.
    """
    start = time.time()
    total = 3 ** n
    
    # Create all vectors
    all_vecs = [_int_to_tuple(i, n) for i in range(total)]
    
    # Define a priority: vectors with more zeros come first (heuristic)
    # Or: vectors that are "more independent" in some sense
    # Let's try: sort by number of zeros descending, then by value
    def prio(v):
        # More zeros = higher priority (processed first)
        zeros = sum(1 for d in v if d == 0)
        # Also prefer vectors with balanced coordinates
        return (-zeros, v)
    
    indices = list(range(total))
    indices.sort(key=lambda i: prio(all_vecs[i]))
    
    S = set()
    forbidden = set()
    
    for idx in indices:
        if time.time() - start > time_limit:
            break
        if idx in forbidden:
            continue
        
        S.add(idx)
        v = all_vecs[idx]
        
        for a_idx in S:
            if a_idx == idx:
                continue
            a = all_vecs[a_idx]
            w = tuple((2 * v[i] - a[i]) % 3 for i in range(n))
            w_idx = _tuple_to_int(w)
            forbidden.add(w_idx)
    
    return S


def priority(vector, n):
    if n not in _PRIORITY_CACHE:
        # Perform search for this n
        time_limit = 250.0  # Leave time for overhead
        
        best_set = set()
        best_size = 0
        
        start_search = time.time()
        seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
        
        # Run multiple strategies with different seeds
        for seed in seeds:
            if time.time() - start_search > time_limit:
                break
            remaining = time_limit - (time.time() - start_search)
            if remaining < 10.0:
                break
            
            # Random greedy
            S = _greedy_cap_set_fast(n, seed, remaining * 0.6)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
            
            # Priority-based greedy
            if time.time() - start_search < time_limit - 10.0:
                remaining2 = time_limit - (time.time() - start_search)
                S2 = _greedy_with_priority(n, seed, remaining2 * 0.5)
                if len(S2) > best_size:
                    best_size = len(S2)
                    best_set = S2
        
        # Convert best_set (indices) to set of tuples
        best_tuples = set()
        for idx in best_set:
            best_tuples.add(_int_to_tuple(idx, n))
        
        _PRIORITY_CACHE[n] = best_tuples
    
    return 1.0 if vector in _PRIORITY_CACHE[n] else 0.0
