import random
import time
from itertools import combinations

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

def max_sidon_size(arr):
    n = len(arr)
    best = 0
    for mask in range(1, 1 << n):
        subset = [arr[i] for i in range(n) if mask & (1 << i)]
        if len(subset) <= best:
            continue
        if is_sidon(subset):
            best = len(subset)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with safety margin
    
    # Restrict search space to increase collision probability
    search_range = 30
    best_set = list(range(12))
    best_val = max_sidon_size(best_set)
    
    random.seed(42)
    
    # Local search with random restarts
    for restart in range(5000): # More restarts
        if time.time() > deadline:
            break
        
        # Random initialization from small range
        current_set = random.sample(range(search_range), 12)
        current_val = max_sidon_size(current_set)
        
        if current_val < best_val:
            best_val = current_val
            best_set = current_set[:]
            if best_val <= 4:  # Very good, stop
                break
        
        # Local search: try swaps with values in the small range
        for step in range(100):
            if time.time() > deadline:
                break
            
            i = random.randint(0, 11)
            new_val = random.randint(0, search_range - 1)
            if new_val in current_set:
                continue
            
            old_val = current_set[i]
            current_set[i] = new_val
            new_val_result = max_sidon_size(current_set)
            
            if new_val_result < current_val:
                current_val = new_val_result
                if current_val < best_val:
                    best_val = current_val
                    best_set = current_set[:]
                    if best_val <= 4:
                        break
            elif new_val_result == current_val and random.random() < 0.2:
                # Accept equal moves to explore
                pass
            else:
                current_set[i] = old_val
            
            if best_val <= 4:
                break
    
    return best_set
