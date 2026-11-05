import numpy as np
import time
import random

def priority(adjacency):
    N = adjacency.shape[0]
    # Convert to list of neighbors
    # adjacency is a bool matrix
    # Using .nonzero() or similar
    neighbors = [list(np.where(adjacency[i])[0]) for i in range(N)]
    
    def simulate(order):
        blocked = set()
        count = 0
        for v in order:
            if v not in blocked:
                count += 1
                blocked.update(neighbors[v])
        return count

    # Initial order: Lowest degree first
    degrees = [len(neighbors[i]) for i in range(N)]
    initial_order = sorted(range(N), key=lambda v: degrees[v])
    
    best_order = initial_order[:]
    best_size = simulate(best_order)
    
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # Local Search
    while time.time() < time_limit:
        current_order = best_order[:]
        current_size = best_size
        no_improve = 0
        max_no_improve = 1000  # Restart if stuck
        
        while time.time() < time_limit and no_improve < max_no_improve:
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            
            # Swap
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
            new_size = simulate(current_order)
            
            if new_size > current_size:
                current_size = new_size
                no_improve = 0
            else:
                # Revert
                current_order[i], current_order[j] = current_order[j], current_order[i]
                no_improve += 1
        
        if current_size > best_size:
            best_size = current_size
            best_order = current_order[:]
        else:
            # If we didn't improve, we might want to perturb the best_order more aggressively
            # or just restart the local search from the same best_order with different random swaps
            # Since we reset current_order to best_order at the start of the outer loop,
            # this effectively does a fresh local search from the global best.
            pass

    # Construct scores
    scores = np.zeros(N)
    for idx, v in enumerate(best_order):
        # Higher score means picked earlier
        scores[v] = N - idx
        
    return scores
