import math
from functools import reduce
import time
import random

def construct():
    # Calculate LCM of 1..24
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def lcm(a, b):
        return a * b // gcd(a, b)
        
    L = reduce(lcm, range(1, 25))
    weights = {i: L // i for i in range(1, 25)}
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Greedy Construction
    reachable = {0}
    selected = []
    
    # Process from largest to smallest (smallest weight to largest weight)
    for x in range(24, 0, -1):
        w_x = weights[x]
        if w_x in reachable:
            continue
        else:
            selected.append(x)
            new_sums = set()
            for s in reachable:
                ns = s + w_x
                # Optional pruning: we don't need sums that are too large?
                # But for future checks, we need all sums.
                # Max sum is L. L is ~5e9. Set size might grow.
                # However, with 24 items, 2^24 is 16M.
                # In practice, the number of distinct sums is much smaller.
                new_sums.add(ns)
            reachable.update(new_sums)
            
    best_set = set(selected)
    best_size = len(best_set)
    
    # 2. Local Search
    def is_valid(S):
        # S is a set of integers
        if not S:
            return True
        S_list = list(S)
        for x in S_list:
            w_x = weights[x]
            others = [y for y in S_list if y != x]
            # Check if w_x is a subset sum of others
            # Use BFS/DFS with pruning
            # To speed up, sort others by weight?
            # We just need to know if w_x is reachable.
            # We can use a set of reachable sums, pruning > w_x
            reachable_sums = {0}
            for y in others:
                w_y = weights[y]
                new_sums = set()
                for s in reachable_sums:
                    ns = s + w_y
                    if ns <= w_x:
                        new_sums.add(ns)
                if not new_sums:
                    break
                reachable_sums.update(new_sums)
                if w_x in reachable_sums:
                    return False
        return True

    def get_neighbors(S):
        # Generate swaps and additions
        S_list = list(S)
        neighbors = []
        
        # Additions
        for y in range(1, 25):
            if y not in S:
                new_S = S.union([y])
                if is_valid(new_S):
                    neighbors.append(new_S)
        
        # Swaps (remove x, add y)
        for x in S_list:
            for y in range(1, 25):
                if y not in S:
                    new_S = S.difference([x]).union([y])
                    if is_valid(new_S):
                        neighbors.append(new_S)
                        
        return neighbors

    current_set = best_set
    current_size = best_size
    
    # Hill Climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        neighbors = get_neighbors(current_set)
        
        # Find the best neighbor (largest size)
        best_neighbor = None
        best_neighbor_size = current_size
        
        for n in neighbors:
            if len(n) > best_neighbor_size:
                best_neighbor_size = len(n)
                best_neighbor = n
                
        if best_neighbor is not None:
            current_set = best_neighbor
            current_size = best_neighbor_size
            improved = True
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            # If we improved, we can continue climbing
            # To avoid getting stuck in local optima, we can randomize or do multiple passes
            # But simple hill climbing is a good start.
            
        # If no improvement via single swap/add, try 2-swap?
        # Or break if no improvement to save time for other strategies?
        # Let's try to break if no single-step improvement to avoid infinite loops on plateaus
        # Or use simulated annealing?
        # For now, simple hill climbing. If stuck, we might miss global max.
        # Let's add a random restart if stuck?
        # Or just rely on the greedy + hill climbing.
        
        # To be more robust, let's try to remove 1 and add 2?
        # This is more expensive.
        # Given N=24, let's stick to 1-swap for speed, but maybe do multiple passes.
        
        if not improved:
             # Try random restarts or perturbations?
             # Let's just break for now to ensure we finish.
             # Or we can try to construct a different greedy solution?
             pass

    return list(best_set)
