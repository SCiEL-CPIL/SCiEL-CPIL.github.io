import time
from math import gcd

def construct():
    deadline = time.time() + 80
    
    # Compute LCM of 1..22
    lcm_val = 1
    for i in range(1, 23):
        lcm_val = lcm_val * i // gcd(lcm_val, i)
    
    c = [lcm_val // (i+1) for i in range(22)]
    
    # Items: (coefficient, original_value)
    # Try sorting descending? Or ascending?
    # Ascending (small coeffs first) was used before.
    # Let's try both or a heuristic. 
    # Actually, let's just use the fast check and a good DFS.
    
    items = sorted([(c[i], i+1) for i in range(22)], key=lambda x: x[0])
    
    best_set = []
    best_size = 0
    
    # Greedy initialization
    current_sums = {0}
    current_set = []
    for coef, val in items:
        new_sums = current_sums | {s + coef for s in current_sums}
        if len(new_sums) == 2 * len(current_sums):
            current_sums = new_sums
            current_set.append(val)
    
    best_set = current_set[:]
    best_size = len(current_set)
    
    def dfs(idx, current, sums):
        nonlocal best_set, best_size
        
        if time.time() > deadline - 2:
            return
            
        # Prune: if even taking all remaining doesn't beat best
        if len(current) + (22 - idx) <= best_size:
            return
            
        # Update best
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            
        # If we have picked all, return
        if idx == 22:
            return
            
        # Try adding each remaining item
        for i in range(idx, 22):
            if time.time() > deadline - 2:
                return
                
            coef, val = items[i]
            
            # Fast collision check using set size
            # If adding coef causes collision, the size of the union of sums and sums+coef
            # will be less than 2 * len(sums).
            # Specifically, if (sums + coef) intersects sums, there is a collision.
            # The new set of sums is sums U (sums + coef).
            # If no collision, size is 2 * len(sums).
            # If collision, size < 2 * len(sums).
            
            # Optimization: Check if coef is in the difference set?
            # No, just compute new_sums.
            
            new_sums = sums | {s + coef for s in sums}
            
            if len(new_sums) == 2 * len(sums):
                dfs(i + 1, current + [val], new_sums)
                
            # Optional: Pruning based on "potential"
            # If we skip this item, can we still beat best?
            # This is handled by the loop continuing.
            
    dfs(0, [], {0})
    
    return best_set
