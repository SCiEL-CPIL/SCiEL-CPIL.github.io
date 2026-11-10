import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if subset is Sidon: all pair sums (i<=j) are unique."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(ambient):
    """Find the size of the largest Sidon subset of ambient."""
    n = len(ambient)
    # Try from largest possible down
    for size in range(n, 1, -1):
        for subset in combinations(ambient, size):
            if is_sidon(subset):
                return size
    return 2  # any 2 distinct elements form a Sidon set

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin
    
    # Start with a reasonable initial set
    # Try to use structure: maybe a set with many 3-term APs
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    # Local search: try to improve by swapping elements
    current = list(range(12))
    current_score = max_sidon_size(current)
    best_set = current[:]
    best_score = current_score
    
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        # Random restart occasionally
        if iteration % 5000 == 0 and iteration > 0:
            current = random.sample(range(129), 12)
            current_score = max_sidon_size(current)
            if current_score < best_score:
                best_set = current[:]
                best_score = current_score
            continue
        
        # Try swapping two elements
        i, j = random.sample(range(12), 2)
        new_elem = random.randint(0, 128)
        
        if new_elem in current:
            continue
            
        old_elem = current[j]
        current[j] = new_elem
        new_score = max_sidon_size(current)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current[:]
        else:
            # Revert
            current[j] = old_elem
    
    return best_set
