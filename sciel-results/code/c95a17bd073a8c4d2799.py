import time
import random
import math
from itertools import combinations

def _is_sidon(subset):
    """Check if a tuple of integers is a Sidon set."""
    sums = set()
    n = len(subset)
    for i in range(n):
        ai = subset[i]
        for j in range(i, n):
            s = ai + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(A):
    """
    Compute the size of the largest subset of A that is a Sidon set.
    A is a list of 12 distinct integers.
    """
    n = len(A)
    # Check subsets in descending order of size
    # We can use itertools.combinations
    for k in range(n, 0, -1):
        for subset in combinations(A, k):
            if _is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit
    
    # Initial guess: Try a few random inits and some heuristics
    best_set = None
    best_score = 13 # Max possible is 12
    
    # Heuristic inits
    candidates = []
    # Random
    for _ in range(50):
        candidates.append(random.sample(range(129), 12))
    # Arithmetic Progressions
    for start in range(0, 100, 10):
        ap = [start + i * 2 for i in range(12)]
        if max(ap) <= 128:
            candidates.append(ap)
    # Dense cluster
    candidates.append(list(range(12)))
    candidates.append(list(range(0, 24, 2)))
    
    for s in candidates:
        score = _max_sidon_size(s)
        if score < best_score:
            best_score = score
            best_set = s
            
    if best_set is None:
        return list(range(12))
        
    current_set = best_set[:]
    current_score = best_score
    
    T = 1.0
    cooling = 0.995
    
    # If we found score 3, we might be done or need to verify if 2 is possible
    # Max Sidon size 2 means no subset of size 3 is Sidon.
    # i.e., every triple has a collision.
    # a+b = c+c or a+a = b+c etc.
    # This is very restrictive.
    
    while time.time() - start_time < time_limit:
        # Generate neighbor
        i = random.randint(0, 11)
        old_val = current_set[i]
        new_val = random.randint(0, 128)
        
        # Ensure distinctness
        if new_val in current_set and new_val != old_val:
            continue
            
        current_set[i] = new_val
        
        new_score = _max_sidon_size(current_set)
        
        # Acceptance criterion
        if new_score < current_score:
            # Better
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            # Worse or equal
            delta = new_score - current_score
            if delta == 0:
                accept = random.random() < 0.5
            else:
                # Avoid overflow in exp
                if delta > 100:
                    accept = False
                else:
                    accept = random.random() < math.exp(-delta / T)
                
            if accept:
                pass
            else:
                # Reject
                current_set[i] = old_val
                continue
                
        current_score = new_score
        T *= cooling
        if T < 0.01:
            T = 0.01

    return best_set
