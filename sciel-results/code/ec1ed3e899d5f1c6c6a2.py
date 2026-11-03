import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0
    
    diffs_per_row = (k + 1) * k // 2
    lb = n_rows * diffs_per_row
    
    # Helper to check time
    def time_up():
        return time.time() - start_time > time_limit - 5

    # Greedy Fallback (Deterministic, fast)
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        break
                    x += 1
                row.append(x)
                used |= new_diffs
            rows.append(row)
        return rows

    # Randomized Greedy Construction
    def randomized_greedy(M, max_restarts=500):
        for restart in range(max_restarts):
            if time_up():
                return None
            
            used = set()
            rows = []
            success = True
            
            for r in range(n_rows):
                if time_up():
                    success = False
                    break
                
                row = [0]
                while len(row) < k + 1:
                    last = row[-1]
                    # Find candidates
                    # x must be > last
                    # x - a not in used for all a in row
                    
                    # Optimization: 
                    # We want x such that {x-a for a in row} is disjoint from used.
                    # This means x not in {a + d for a in row, d in used}
                    
                    # Let's find the smallest few valid candidates
                    # We can iterate x from last+1 upwards
                    # To randomize, we pick a random x from a range [last+1, last+window]
                    # Or we find all valids in a small range and pick one.
                    
                    candidates = []
                    # Limit search window to keep it fast
                    # If M is the target max, we shouldn't exceed M.
                    # But during construction, we just need *a* valid x.
                    # We will check if max(row) <= M at the end? 
                    # No, we want to construct with max <= M.
                    # So x must be <= M.
                    
                    # Find valid x in [last+1, M]
                    # To speed up, we check sequentially but stop after finding a few?
                    # Or just find the first valid one (deterministic) and randomize the ROW ORDER?
                    # No, randomizing value choice is better.
                    
                    # Let's find up to `window_size` valid candidates
                    window_size = 5
                    found = []
                    x = last + 1
                    while x <= M and len(found) < window_size:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            found.append(x)
                        x += 1
                        
                    if not found:
                        success = False
                        break
                    
                    # Pick randomly from found
                    x_choice = random.choice(found)
                    new_diffs = {x_choice - a for a in row}
                    row.append(x_choice)
                    used |= new_diffs
                
                if not success:
                    break
                rows.append(row)
                
                # Optional: Check if we are already exceeding M?
                # The loop ensures x <= M.
                
            if success:
                return rows
        return None

    # Try to find a solution with minimal M
    # Start from LB
    best_rows = None
    
    # We can't iterate M up to LB+200 if randomized_greedy is slow.
    # randomized_greedy is O(n_rows * k * window * check_cost).
    # check_cost is O(k) for set intersection.
    # So it's fast.
    
    # Strategy:
    # 1. Try M = LB, LB+1, ... 
    # 2. For each M, try randomized_greedy with some restarts.
    # 3. If found, return.
    
    # To be safe, if we don't find one quickly, fall back to greedy_construct 
    # which always works but might have large M.
    # But greedy_construct doesn't take M. It just finds *a* solution.
    # We can modify greedy_construct to take M? No, it finds the smallest possible greedily.
    
    # Let's try M from LB to LB + 50.
    for M in range(lb, lb + 50):
        if time_up():
            break
        
        rows = randomized_greedy(M, max_restarts=100)
        if rows:
            return rows
            
    # Fallback
    return greedy_construct()
