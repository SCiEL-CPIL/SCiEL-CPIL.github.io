import time
import random

def construct():
    deadline = time.time() + 85.0
    best_set = []
    best_len = 0
    
    # Helper to run greedy with a specific order of candidates
    def run_greedy(candidates, time_limit):
        selected = []
        existing_sums = set()
        
        for x in candidates:
            if time.time() > time_limit:
                break
            if len(selected) >= 80: # Max possible
                break
            
            # Quick check: if 3*x is already in existing_sums, skip
            if 3 * x in existing_sums:
                continue
                
            # Check all new sums involving x, with early termination
            valid = True
            new_sums_set = set()
            
            # (x, x, x)
            s = 3 * x
            if s in existing_sums or s in new_sums_set:
                valid = False
            else:
                new_sums_set.add(s)
            
            if valid:
                for i, y in enumerate(selected):
                    # (x, x, y)
                    s = 2 * x + y
                    if s in existing_sums or s in new_sums_set:
                        valid = False
                        break
                    new_sums_set.add(s)
                    
                    # (x, y, y)
                    s = x + 2 * y
                    if s in existing_sums or s in new_sums_set:
                        valid = False
                        break
                    new_sums_set.add(s)
                    
                    # (x, y, z) for z > y
                    for z in selected[i+1:]:
                        s = x + y + z
                        if s in existing_sums or s in new_sums_set:
                            valid = False
                            break
                        new_sums_set.add(s)
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                existing_sums.update(new_sums_set)
        
        return selected

    # Strategy 1: Natural order
    candidates_natural = list(range(1, 4097))
    set_natural = run_greedy(candidates_natural, deadline)
    if len(set_natural) > best_len:
        best_set = set_natural
        best_len = len(set_natural)
        
    # Strategy 2: Randomized restarts
    # Use remaining time for random shuffles
    while time.time() < deadline - 5.0:
        candidates_random = list(range(1, 4097))
        random.shuffle(candidates_random)
        
        set_random = run_greedy(candidates_random, deadline)
        
        if len(set_random) > best_len:
            best_set = set_random
            best_len = len(set_random)
            
        # Early termination if we hit a theoretical upper bound estimate
        # Max size for B_3[1] in [1, N] is approx N^(1/3). 4096^(1/3) = 16.
        # However, constants matter. Let's say 20 is a safe high bound.
        if best_len >= 20:
            break

    return best_set
