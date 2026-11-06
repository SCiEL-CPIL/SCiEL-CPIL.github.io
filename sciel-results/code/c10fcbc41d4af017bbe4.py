import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    N = 1000
    best_set = []
    best_size = 0
    
    def is_valid_addition(x, chosen, counts):
        # Check if adding x to chosen violates the constraint
        # New sums: x + y for y in chosen, and 2*x
        # We need to ensure that for all s in new_sums, count[s] + 1 <= 2
        # i.e., count[s] < 2 for all s in new_sums
        for y in chosen:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        if counts.get(2 * x, 0) >= 2:
            return False
        return True

    def add_to_set(x, chosen, counts):
        for y in chosen:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        counts[2 * x] = counts.get(2 * x, 0) + 1
        chosen.append(x)

    # Initial deterministic greedy to get a baseline
    chosen = []
    counts = {}
    for x in range(1, N + 1):
        if is_valid_addition(x, chosen, counts):
            add_to_set(x, chosen, counts)
    best_set = chosen.copy()
    best_size = len(chosen)
    
    # Randomized restarts
    while time.time() < deadline:
        random.shuffle_order = list(range(1, N + 1))
        random.shuffle(random.shuffle_order)
        
        chosen = []
        counts = {}
        for x in random.shuffle_order:
            if is_valid_addition(x, chosen, counts):
                add_to_set(x, chosen, counts)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen.copy()
            
    return best_set
