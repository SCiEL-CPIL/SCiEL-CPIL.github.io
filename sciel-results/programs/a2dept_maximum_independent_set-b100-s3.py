import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    # Precompute neighbor lists for fast access
    # adjacency is a bool matrix
    # Convert to list of lists of neighbors
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    
    def greedy_mis_size(order):
        """
        order: list of vertices in the order they should be considered (first is highest priority)
        Returns the size of the independent set.
        """
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                # Mark neighbors as selected (blocked)
                for u in neighbors[v]:
                    selected[u] = True
        return count

    # Initial order: Minimum Degree Heuristic (sort by -degree)
    degrees = adjacency.sum(axis=1)
    # We want low degree first, so sort by -degree descending? 
    # No, sort by degree ascending.
    # The decoder picks highest score first.
    # So we want the best vertices to have the highest scores.
    # If we use Min-Degree heuristic, low degree vertices are good.
    # So score should be high for low degree.
    # Let's just create the order directly.
    
    # Initial order: vertices sorted by degree (ascending)
    initial_order = list(np.argsort(degrees))
    
    current_order = initial_order
    current_size = greedy_mis_size(current_order)
    
    best_order = current_order[:]
    best_size = current_size
    
    # Local Search: Simulated Annealing or Hill Climbing on Permutations
    # We will use a simple random swap local search with restarts.
    
    # To speed up, we can use a "fast" greedy check.
    # But n=200 is small enough that full greedy check is fast.
    # 200 * 200 = 40000 operations per check.
    # We can do ~100,000 checks in 1 second.
    # 140 seconds -> ~14,000,000 checks.
    # This is plenty for local search.
    
    # We will do multiple restarts with different initial orders.
    # 1. Min Degree
    # 2. Random
    # 3. Max Degree (bad, but maybe for some graphs?)
    # 4. Random with noise on degree
    
    rng = np.random.default_rng(42)
    
    num_restarts = 10
    steps_per_restart = 5000
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if r == 0:
            order = initial_order[:]
        elif r == 1:
            order = list(rng.permutation(n))
        elif r == 2:
            # Max degree
            order = list(np.argsort(-degrees))
        else:
            # Noisy degree
            noisy_deg = degrees + rng.normal(0, 1, n)
            order = list(np.argsort(noisy_deg))
            
        current_size = greedy_mis_size(order)
        
        if current_size > best_size:
            best_size = current_size
            best_order = order[:]
            
        # Local Search Phase
        # We will try swaps.
        # To make it faster, we can just do random swaps and keep if improved (Hill Climbing).
        # Or Simulated Annealing to escape local optima.
        # Given the budget, Hill Climbing with restarts is usually robust.
        
        temp = 1.0
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            # Pick two random indices to swap
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
                
            order[i], order[j] = order[j], order[i]
            
            new_size = greedy_mis_size(order)
            
            if new_size >= current_size:
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_order = order[:]
            else:
                # Revert
                order[i], order[j] = order[j], order[i]
                
                # Optional: Simulated Annealing acceptance
                # delta = current_size - new_size (positive)
                # if rng.random() < np.exp(-delta / temp):
                #    keep
                # But MIS size is integer, so delta is at least 1.
                # exp(-1/temp) is small.
                # Let's stick to strict improvement for simplicity and speed, 
                # relying on restarts for diversity.
                
    # Convert best_order to scores
    # The decoder sorts by descending score.
    # So the first element in best_order should have the highest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
