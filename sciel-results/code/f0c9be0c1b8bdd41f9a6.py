import random
import itertools
import time

def is_sidon(S):
    """Check if set S is a Sidon set (all unordered pair sums unique including diagonals)."""
    sums = set()
    for i, x in enumerate(S):
        for j, y in enumerate(S):
            if i <= j:
                s = x + y
                if s in sums:
                    return False
                sums.add(s)
    return True

def has_sidon_subset_of_size_k(A, k):
    """Check if A contains a Sidon subset of size k."""
    # If k is small, we can iterate through combinations
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave margin for evaluator overhead
    
    # Try to find a set with no Sidon subset of size 5
    # If found, the max Sidon size is <= 4.
    # Baseline is likely 5.
    
    # Heuristic: Random search
    # We can also try to construct it.
    # A set of multiples of a large number might help?
    # Or a set that is "dense" in a small range?
    # range(12) has max Sidon 5.
    # Let's try random sets.
    
    best_set = list(range(12))
    best_max_size = 5 # Assume baseline is 5
    
    # First, verify baseline
    if not has_sidon_subset_of_size_k(best_set, 5):
        best_max_size = 4
        # Check if 4 is possible
        if not has_sidon_subset_of_size_k(best_set, 4):
            best_max_size = 3
            
    # Random search for better sets
    for _ in range(10000):
        if time.time() > deadline:
            break
        
        A = random.sample(range(129), 12)
        
        # Check for Sidon subset of size 5
        if has_sidon_subset_of_size_k(A, 5):
            continue
            
        # If no Sidon subset of size 5, max size is <= 4
        # Check if it has Sidon subset of size 4
        if has_sidon_subset_of_size_k(A, 4):
            # Max size is 4
            return A
        else:
            # Max size is <= 3
            return A

    return best_set
