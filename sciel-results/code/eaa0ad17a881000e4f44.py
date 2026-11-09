import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Precompute neighbor lists and masks for fast evaluation
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    degrees = np.array([len(nei) for nei in neighbors])
    
    # Precompute neighbor masks as integer arrays for vectorized evaluation
    # For each vertex v, neighbor_mask[v] is a boolean array of length n
    # where neighbor_mask[v][u] = True if u is a neighbor of v
    neighbor_masks = np.zeros((n, n), dtype=bool)
    for i in range(n):
        if len(neighbors[i]) > 0:
            neighbor_masks[i, neighbors[i]] = True
    
    def greedy_mis_size(order):
        """
        order: list of vertices in the order they should be considered (first is highest priority)
        Returns the size of the independent set.
        Uses vectorized operations with precomputed neighbor masks.
        """
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                # Mark neighbors as blocked using precomputed mask
                if len(neighbors[v]) > 0:
                    selected |= neighbor_masks[v]
        return count
    
    # Initial Order: Min-Degree (Low degree = High Priority = High Score)
    initial_order = list(np.argsort(degrees))
    
    current_order = initial_order
    current_size = greedy_mis_size(current_order)
    
    best_order = current_order[:]
    best_size = current_size
    
    rng = np.random.default_rng(42)
    
    # Local Search: Simulated Annealing on the permutation
    num_restarts = 5
    steps_per_restart = 20000
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if r == 0:
            order = initial_order[:]
        else:
            # Perturb the best order found so far
            order = best_order[:]
            # Perform a few random swaps to escape local optimum
            for _ in range(10):
                i, j = rng.integers(0, n, 2)
                order[i], order[j] = order[j], order[i]
            
        current_size = greedy_mis_size(order)
        
        if current_size > best_size:
            best_size = current_size
            best_order = order[:]
            
        temp = 2.0
        cooling = 0.9995
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            # Pick two random indices to swap
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
                
            order[i], order[j] = order[j], order[i]
            
            new_size = greedy_mis_size(order)
            
            delta = current_size - new_size  # Positive if we got worse
            
            if delta < 0:
                # Improvement
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_order = order[:]
            elif delta == 0:
                # Neutral
                current_size = new_size
            else:
                # Worse
                # Accept with probability exp(-delta / temp)
                if rng.random() < np.exp(-delta / temp):
                    current_size = new_size
                else:
                    # Revert
                    order[i], order[j] = order[j], order[i]
                    
            temp *= cooling
            if temp < 0.01:
                break
    
    # Final polish: try targeted local improvements on the best order
    if time.time() < deadline - 2:
        polish_order = best_order[:]
        polish_size = best_size
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(n - 1):
                if time.time() > deadline - 1:
                    break
                # Try swapping adjacent elements
                j = i + 1
                polish_order[i], polish_order[j] = polish_order[j], polish_order[i]
                new_size = greedy_mis_size(polish_order)
                if new_size > polish_size:
                    polish_size = new_size
                    improved = True
                    if polish_size > best_size:
                        best_size = polish_size
                        best_order = polish_order[:]
                else:
                    # Revert
                    polish_order[i], polish_order[j] = polish_order[j], polish_order[i]
    
    # Convert best_order to scores
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
    
    return scores
