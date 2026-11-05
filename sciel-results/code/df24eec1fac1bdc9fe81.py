import time
import math
import random
from itertools import combinations

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    # Universe: 1..22
    universe = list(range(1, 23))
    
    # Compute LCM of 1..22 to scale reciprocals to integers
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for i in range(1, 23):
        L = lcm(L, i)
        
    # Map each element a to its integer value L/a
    values = {a: L // a for a in universe}
    
    best_size = 0
    best_set = []
    
    # We will perform DFS.
    # To make it faster, we can try to find a large set quickly.
    # Greedy initialization to get a good lower bound.
    
    def greedy_init(order):
        current = []
        current_sums = {0}
        for a in order:
            v = values[a]
            new_sums = {s + v for s in current_sums}
            if not current_sums.intersection(new_sums):
                current.append(a)
                current_sums.update(new_sums)
        return current, current_sums

    # Try a few greedy orders to get a good initial best
    random.seed(42)
    orders = [
        universe,
        universe[::-1],
        sorted(universe, key=lambda x: values[x]),
        sorted(universe, key=lambda x: -values[x]),
    ]
    for _ in range(10):
        shuffled = universe[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    for order in orders:
        if time.time() > deadline:
            break
        g_set, g_sums = greedy_init(order)
        if len(g_set) > best_size:
            best_size = len(g_set)
            best_set = g_set[:]
            
    # DFS to find the maximum
    # We search for sets of size > best_size
    
    def dfs(available, current_set, current_sums, current_size):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
        # Pruning: even if we add all remaining, we can't beat best
        if current_size + len(available) <= best_size:
            return
            
        if not available:
            return
            
        # Heuristic: try to include elements that are "large" first?
        # Or just iterate through available.
        # To allow finding different sets, we can shuffle available if needed,
        # but deterministic order is better for reproducibility and pruning.
        
        # We iterate through available elements.
        # We can try to include 'a' or exclude 'a'.
        # To find the MAXIMUM size, we should prioritize inclusion.
        
        # Optimization: If we find a set of size N, we only care about finding N+1.
        
        for i, a in enumerate(available):
            if time.time() > deadline:
                return
            
            v = values[a]
            # Check if adding a causes collision
            # New sums are current_sums U (current_sums + v)
            # Collision if (current_sums + v) intersects current_sums
            
            # Efficient check:
            # any((s + v) in current_sums for s in current_sums)
            # Since current_sums is a set, this is fast for small sizes.
            
            has_collision = False
            # We can break early if collision found
            for s in current_sums:
                if (s + v) in current_sums:
                    has_collision = True
                    break
                    
            if not has_collision:
                new_sums = current_sums | {s + v for s in current_sums}
                # Recurse with 'a' included
                # Pass a copy of current_set and new_sums
                # To save memory/time, we can pass references and backtrack, 
                # but sets are not easily mutable in a way that supports O(1) undo 
                # unless we track added elements. 
                # Given small sizes, copying is acceptable.
                
                dfs(available[i+1:], current_set + [a], new_sums, current_size + 1)
            
            # Recurse with 'a' excluded
            # We only need to explore exclusion if inclusion didn't lead to a solution 
            # larger than current best? No, we need to explore all branches to be sure.
            # However, if we already found a set of size best_size, and we are 
            # looking for larger, we can prune branches that can't exceed best_size.
            
            dfs(available[i+1:], current_set, current_sums, current_size)
            
            # If we found a set larger than best_size in the include branch,
            # we might still need to check exclude branches for even larger sets?
            # Yes, because excluding 'a' might allow adding two other elements later.
            # So we always explore both branches unless pruned by size bound.

    # Start DFS
    # Order of available matters. 
    # Let's try starting with the greedy best set's complement?
    # No, standard DFS from empty set with a good ordering.
    
    # Let's use the universe sorted by value descending (smaller a first) 
    # as the initial available list for DFS.
    # Actually, mixing in some randomness or trying different start orders helps.
    
    # Since the search space is 2^22 ~ 4M, and we prune heavily, 
    # a single well-ordered DFS might be enough.
    
    # Let's try DFS with universe sorted by value descending (1, 2, 3...)
    # This tends to add "heavy" elements first.
    
    start_available = sorted(universe, key=lambda x: values[x], reverse=True)
    
    # To increase robustness, we can run multiple DFS passes with different orderings
    # if time permits.
    
    dfs_orders = [
        sorted(universe, key=lambda x: values[x], reverse=True), # Desc value (1,2,3..)
        sorted(universe, key=lambda x: values[x]),               # Asc value (22,21..)
        sorted(universe, key=lambda x: x),                       # Natural
        sorted(universe, key=lambda x: -x),                      # Reverse natural
    ]
    
    # Add a few random shuffles
    for _ in range(5):
        shuffled = universe[:]
        random.shuffle(shuffled)
        dfs_orders.append(shuffled)
        
    for order in dfs_orders:
        if time.time() > deadline:
            break
        # Reset best if we want to find the absolute max? 
        # No, we keep the global best.
        # But DFS is deterministic for a given order.
        # If we already found size 10, and we run another DFS, it will prune 
        # branches that can't exceed 10. This is good.
        
        dfs(order, [], {0}, 0)
        
    return best_set

# Quick sanity check logic for the collision check in DFS:
# current_sums contains all subset sums of the current set.
# If we add v, the new subset sums are S U (S+v).
# If there is an x in S and y in S such that x = y + v, then x-v = y.
# This means v is a difference between two elements in S.
# Wait, the condition for distinct subset sums is that no non-trivial linear combination 
# with coefficients in {-1, 0, 1} sums to 0.
# Adding v causes a collision if there exists a subset A of current set such that 
# sum(A) = sum(B) + v, where B is a subset of current set.
# This is equivalent to sum(A \ B) - sum(B \ A) = v.
# The check `any((s + v) in current_sums for s in current_sums)` is correct.
# It checks if s + v is already a subset sum.
# If s + v = t where t is a subset sum, then sum(B) + v = sum(A).
# So v = sum(A) - sum(B).
# This corresponds to the linear combination sum(A) - sum(B) - v = 0.
# Coefficients are 1 for A\B, -1 for B\A, and -1 for v.
# This is a valid non-trivial relation if A != B or if v is not in the set (which it isn't yet).
# So the check is correct.

