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
        
        # Strategy: Replace one element with a random one, or swap two elements
        if random.random() < 0.5:
            # Replace one element
            idx = random.randint(0, 11)
            old_val = new_set[idx]
            new_val = random.randint(0, 128)
            
            # Ensure distinctness
            if new_val in new_set and new_val != old_val:
                continue
                
            new_set[idx] = new_val
        else:
            # Swap two elements with random values
            idx1, idx2 = random.sample(range(12), 2)
            val1 = random.randint(0, 128)
            val2 = random.randint(0, 128)
            
            # Ensure distinctness
            if val1 == val2:
                continue
            if val1 in new_set and val1 != new_set[idx1]:
                continue
            if val2 in new_set and val2 != new_set[idx2]:
                continue
            if (val1 in new_set and val1 == new_set[idx2]) or (val2 in new_set and val2 == new_set[idx1]):
                continue
                
            new_set[idx1] = val1
            new_set[idx2] = val2
            
        new_set.sort()
        
        new_score = max_sidon_size(new_set)
        
        if new_score < best_score:
            best_score = new_score
            best_set = new_set[:]
            current_set = new_set[:]
            current_score = new_score
            if best_score <= 3:
                break
        else:
            # Accept with some probability to escape local minima
            if random.random() < 0.1:
                current_set = new_set[:]
                current_score = new_score
    
    return best_set
