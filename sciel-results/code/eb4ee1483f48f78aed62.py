import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # LCM of 1..22
    # L = 232792560
    L = 232792560
    
    # Weights w_i = L / i
    weights = [L // i for i in range(1, 23)]
    
    best_set = []
    best_size = 0
    
    # Greedy: Try to add elements in different orders
    # Order 1: Ascending i (1 to 22) - Large weights first
    # Order 2: Descending i (22 to 1) - Small weights first
    # Order 3: Randomized?
    
    def try_greedy(order):
        current_set = []
        sums = {0}
        for i in order:
            w = weights[i-1]
            new_sums = set(sums)
            for s in sums:
                ns = s + w
                if ns in new_sums:
                    break
                new_sums.add(ns)
            else:
                # No collision
                current_set.append(i)
                sums = new_sums
        return current_set
    
    # Try ascending
    set_asc = try_greedy(range(1, 23))
    if len(set_asc) > best_size:
        best_size = len(set_asc)
        best_set = set_asc
        
    # Try descending
    set_desc = try_greedy(range(22, 0, -1))
    if len(set_desc) > best_size:
        best_size = len(set_desc)
        best_set = set_desc
        
    # If we found 16, we are likely done.
    # If not, we might need a smarter search.
    # But given the observation of 16, greedy might find it.
    
    # Let's also try a "skip one" heuristic if greedy fails to reach 16
    # But for now, return the best found.
    
    if time.time() > deadline:
        return best_set
        
    return best_set
