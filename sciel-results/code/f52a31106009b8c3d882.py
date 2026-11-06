import random
import time
from math import lcm
from functools import reduce

# LCM of 1..22
L = reduce(lcm, range(1, 23))
# Values v_i = L // i
V = [L // i for i in range(1, 23)]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin for 90s limit
    
    # Seed set: [1, 2, 3, 5, 7, 11, 13, 17, 19] -> indices 0, 1, 2, 4, 6, 10, 12, 16, 18
    best_set = [0, 1, 2, 4, 6, 10, 12, 16, 18]
    best_size = len(best_set)
    
    all_indices = list(range(22))
    
    def indices_to_numbers(indices):
        return [i + 1 for i in indices]
    
    # Random greedy construction
    for _ in range(50000): # High number, but breaks on time
        if time.time() > deadline:
            break
        
        random.shuffle(all_indices)
        current_set = []
        current_sums = {0}
        
        for idx in all_indices:
            val = V[idx]
            valid = True
            new_sums = set()
            # Check for collisions
            for s in current_sums:
                ns = s + val
                if ns in current_sums:
                    valid = False
                    break
                new_sums.add(ns)
            
            if valid:
                current_set.append(idx)
                current_sums.update(new_sums)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return indices_to_numbers(best_set)
