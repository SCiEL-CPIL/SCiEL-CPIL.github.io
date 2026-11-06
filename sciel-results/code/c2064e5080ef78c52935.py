import random
import time
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave some buffer for 60s limit
    
    best_set = []
    best_size = 0
    
    # Initial deterministic run to get a baseline
    # The original greedy from 1 upwards
    def run_greedy(order):
        selected = []
        current_sums = set()
        
        for x in order:
            # Generate new sums involving x
            new_sums = set()
            
            # x + x + x
            s = 3 * x
            if s in current_sums:
                continue
            new_sums.add(s)
            
            # x + x + a
            conflict = False
            for a in selected:
                s = 2 * x + a
                if s in current_sums or s in new_sums:
                    conflict = True
                    break
                new_sums.add(s)
            if conflict:
                continue
                
            # x + a + b
            for a, b in combinations_with_replacement(selected, 2):
                s = x + a + b
                if s in current_sums or s in new_sums:
                    conflict = True
                    break
                new_sums.add(s)
            
            if not conflict:
                selected.append(x)
                current_sums.update(new_sums)
                
        return selected

    # 1. Deterministic greedy from 1..4096
    baseline = run_greedy(range(1, 4097))
    best_set = baseline
    best_size = len(baseline)
    
    # 2. Randomized restarts
    # We can shuffle the range and try to pack more.
    # Sometimes starting with larger numbers or random numbers helps?
    # Actually, for Sidon sets, smaller numbers are more "expensive" in terms of sum space usage?
    # No, smaller numbers have fewer sum collisions?
    # Let's just shuffle.
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        # Shuffle the candidate range
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        current_set = run_greedy(candidates)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # If we found a very large set, we might stop early?
        # Theoretical max is ~41. If we hit 41, we can stop.
        if best_size >= 41:
            break
            
    return best_set
