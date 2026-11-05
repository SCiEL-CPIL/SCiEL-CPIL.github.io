import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Initial state: random 32 distinct integers in 1..512
    current_set = set(random.sample(range(1, 513), 32))
    current_score = _score(current_set)
    best_set = set(current_set)
    best_score = current_score
    
    # Simulated Annealing / Local Search
    T = 100.0  # Initial temperature
    cooling = 0.995
    
    while time.time() < deadline:
        # Generate a neighbor: pick 1-2 elements to move or swap
        if random.random() < 0.5:
            # Move one element
            idx = random.randint(0, 31)
            elem = list(current_set)[idx]
            new_elem = random.randint(1, 512)
            if new_elem not in current_set:
                new_set = set(current_set)
                new_set.remove(elem)
                new_set.add(new_elem)
            else:
                continue
        else:
            # Swap two elements (one in, one out)
            idx = random.randint(0, 31)
            elem_out = list(current_set)[idx]
            elem_in = random.randint(1, 512)
            if elem_in not in current_set:
                new_set = set(current_set)
                new_set.remove(elem_out)
                new_set.add(elem_in)
            else:
                continue
        
        new_score = _score(new_set)
        
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            if new_score > best_score:
                best_set = new_set
                best_score = new_score
        else:
            # Accept with probability based on temperature
            delta = new_score - current_score
            if random.random() < np.exp(delta / T):
                current_set = new_set
                current_score = new_score
        
        T *= cooling
        if T < 0.1:
            T = 0.1
            
        # Periodic restart if stuck? Or just let it cool.
        # Given the time budget, simple SA is sufficient.
        
    return list(best_set)

def _score(s):
    """Calculate the number of distinct d > 0 such that a, a+d, a+2d are in s."""
    s = set(s)
    diffs = set()
    # Optimize: iterate over pairs (a, b) where a < b
    # d = b - a
    # Check if a + 2d = b + d is in s
    elements = sorted(s)
    for i in range(len(elements)):
        a = elements[i]
        for j in range(i + 1, len(elements)):
            b = elements[j]
            d = b - a
            c = b + d
            if c in s:
                diffs.add(d)
    return len(diffs)
# EVOLVE-BLOCK-END
