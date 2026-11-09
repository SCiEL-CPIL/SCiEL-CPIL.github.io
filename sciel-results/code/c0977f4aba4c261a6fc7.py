import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

# Precompute LCM of 1..24
L = 1
for i in range(1, 25):
    L = _lcm(L, i)

vals = [0] * 25
for i in range(1, 25):
    vals[i] = L // i

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_size = 0
    best_set = set()
    
    # Process 24 down to 1
    # current_set: set of selected elements
    # reachable_sums: set of all subset sums of vals[y] for y in current_set
    # We only care about sums <= L (max target)
    
    current_set = set()
    reachable_sums = {0}
    
    def solve(x):
        nonlocal best_size, best_set, current_set, reachable_sums
        
        if time.time() > deadline:
            return
            
        # Prune: if current size + remaining <= best, return
        # Remaining elements are 1..x
        if len(current_set) + x <= best_size:
            return
            
        if x == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
            
        # Option 1: Include x
        # Check if vals[x] is in reachable_sums
        if vals[x] not in reachable_sums:
            current_set.add(x)
            
            # Update reachable_sums
            # We need to add vals[x] + s for all s in reachable_sums
            # But we should cap at L? vals[1] = L.
            # Any sum > L is useless for future checks (targets are <= L).
            old_reachable = reachable_sums
            new_sums = set()
            for s in old_reachable:
                ns = s + vals[x]
                if ns <= L:
                    new_sums.add(ns)
            reachable_sums = old_reachable.union(new_sums)
            
            solve(x - 1)
            
            # Backtrack
            reachable_sums = old_reachable
            current_set.remove(x)
            
        # Option 2: Exclude x
        solve(x - 1)

    solve(24)
    
    if not best_set:
        best_set = {24}
        
    return sorted(best_set)
