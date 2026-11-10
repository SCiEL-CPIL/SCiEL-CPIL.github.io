import itertools
import random
import time

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sidon_subsets_size_5(S):
    count = 0
    for subset in itertools.combinations(S, 5):
        if is_sidon(subset):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # We aim for 0 Sidon subsets of size 5.
    # If we find one, max size is <= 4.
    
    best_set = list(range(12))
    best_count = count_sidon_subsets_size_5(best_set)
    
    # Random search
    while time.time() < deadline:
        # Try small ranges
        # Range size 20-40
        r = random.randint(20, 40)
        S = random.sample(range(r), 12)
        
        c = count_sidon_subsets_size_5(S)
        
        if c == 0:
            return S
            
        if c < best_count:
            best_count = c
            best_set = S
            
    return best_set
