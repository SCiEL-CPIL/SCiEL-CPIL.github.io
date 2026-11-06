import time
import random
from itertools import combinations

def is_sidon_with_diagonals(subset):
    """
    Check if a subset of integers has all unordered pair sums (including diagonals) unique.
    """
    sums = set()
    k = len(subset)
    for i in range(k):
        for j in range(i, k):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(candidate_set):
    """
    Find the size of the largest subset of candidate_set that is Sidon with diagonals.
    candidate_set is a list of 12 distinct integers.
    """
    n = len(candidate_set)
    # Check from largest possible size down to 1
    # Max size is n
    for k in range(n, 0, -1):
        found = False
        # Iterate over all combinations of size k
        for comb in combinations(candidate_set, k):
            if is_sidon_with_diagonals(comb):
                found = True
                break
        if found:
            return k
    return 0

def construct():
    start_time = time.time()
    deadline = 85.0 # 85 seconds to be safe
    
    # Initial guess: Arithmetic Progression 0..11
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    # Local search
    # We try to swap elements or perturb them to reduce the max Sidon size.
    # Since the search space is discrete and small (12 elements), we can try random restarts or hill climbing.
    
    # Heuristic: APs are bad for Sidon (many collisions), so they might have small max Sidon size.
    # Let's check the score of 0..11 first.
    # If best_score is already 3 or 4, it's good.
    
    # Try to improve by local search
    current_set = best_set[:]
    current_score = best_score
    
    while time.time() < deadline:
        # Try a random swap or perturbation
        new_set = current_set[:]
        # Pick two indices to swap or modify
        i, j = random.sample(range(12), 2)
        
        # Strategy 1: Swap with a random number in [0, 128]
        # Actually, just swapping two elements in the set doesn't change the set.
        # We need to change the values.
        # Let's try replacing one element with a random one.
        idx = random.randint(0, 11)
        old_val = new_set[idx]
        new_val = random.randint(0, 128)
        
        # Ensure distinctness
        if new_val in new_set and new_val != old_val:
            continue
            
        new_set[idx] = new_val
        new_set.sort()
        
        new_score = max_sidon_size(new_set)
        
        if new_score < best_score:
            best_score = new_score
            best_set = new_set[:]
            current_set = new_set[:]
            current_score = new_score
            if best_score <= 3:
                # 3 is very good. 2 is even better but unlikely for 12 distinct ints in [0,128]?
                # Actually, any 2 distinct ints form a Sidon set. So min is 2.
                # Can we get max size 2?
                # That means every subset of size 3 has a collision.
                # a+b = c+d? No, for size 3 {a,b,c}, sums are 2a, a+b, 2b, a+c, b+c, 2c.
                # Collision if a+c=2b (AP).
                # So if every triple is an AP, max size is 2.
                # Can we have 12 distinct ints in [0,128] such that every triple is an AP?
                # No. {0, 1, 2} is AP. {0, 1, 3} is not.
                # It's impossible to have every triple as an AP for n>3.
                # So max size 3 is likely the minimum achievable.
                break
        else:
            # Accept with some probability or just reject?
            # Simple hill climbing: reject if not better.
            # To escape local minima, we can occasionally accept worse or equal.
            if random.random() < 0.1:
                current_set = new_set[:]
                current_score = new_score
    
    return best_set
