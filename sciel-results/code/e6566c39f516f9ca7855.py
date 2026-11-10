"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute neighbors (conflicts)
    neighbors = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                neighbors[i].add(j)
                neighbors[j].add(i)
    
    best_set = set()
    best_size = 0
    
    # Randomized Greedy with Set Intersection
    # 10000 restarts should be fast enough with set intersection
    for _ in range(10000):
        random.shuffle(numbers)
        current_set = set()
        
        for num in numbers:
            if not (neighbors[num] & current_set):
                current_set.add(num)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return list(best_set)
# EVOLVE-BLOCK-END
