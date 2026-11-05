import itertools
import random
import time

def is_sidon(s):
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            total = s[i] + s[j]
            if total in sums:
                return False
            sums.add(total)
    return True

def max_sidon_size(s):
    # Exhaustive search for max Sidon subset
    n = len(s)
    for k in range(n, 2, -1):
        for subset in itertools.combinations(s, k):
            if is_sidon(subset):
                return k
    return 2

def construct():
    # We want to minimize the maximum Sidon subset size.
    # A set with many additive collisions will have a smaller max Sidon subset.
    # 
    # Consider a set with many elements forming arithmetic progressions or 
    # having many repeated differences.
    # 
    # Let's try to construct a set where many pairs sum to the same value.
    # For example, if we have pairs (a, b) and (c, d) such that a+b = c+d,
    # that's a collision.
    # 
    # A classic example of a set with small Sidon number is one that is "additively dense".
    # 
    # Let's try a set like:
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 -> max Sidon size is 5?
    # Let's check a different structure.
    # 
    # What if we take two interleaved arithmetic progressions?
    # Or what if we take a set with many elements close together?
    # 
    # Let's do a quick local search to find a set with smaller max Sidon size.
    # We'll start with a random set and try to swap elements to reduce the max Sidon size.
    
    start_time = time.time()
    timeout = 30  # seconds
    
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    # Try a few random initial sets
    for trial in range(50):
        if time.time() - start_time > timeout:
            break
            
        # Generate a random set of 12 distinct integers in [0, 128]
        current_set = random.sample(range(129), 12)
        current_score = max_sidon_size(current_set)
        
        if current_score < best_score:
            best_set = current_set.copy()
            best_score = current_score
        
        # Local search: try swapping one element
        for _ in range(100):
            if time.time() - start_time > timeout:
                break
            
            i = random.randint(0, 11)
            old_val = current_set[i]
            new_val = random.randint(0, 128)
            
            if new_val in current_set and new_val != old_val:
                continue
                
            current_set[i] = new_val
            new_score = max_sidon_size(current_set)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_set = current_set.copy()
                    best_score = new_score
            else:
                current_set[i] = old_val
    
    return best_set
