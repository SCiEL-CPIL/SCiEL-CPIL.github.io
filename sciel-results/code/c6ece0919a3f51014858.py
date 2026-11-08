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
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    degrees = adjacency.sum(axis=1)
    
    def greedy_mis_size(order):
        """Fast greedy MIS size computation."""
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not blocked[v]:
                count += 1
                blocked[v] = True
                nbrs = neighbors[v]
                if len(nbrs) > 0:
                    blocked[nbrs] = True
        return count
    
    # Initial order: Minimum Degree
    initial_order = list(np.argsort(degrees))
    
    best_order = initial_order[:]
    best_size = greedy_mis_size(best_order)
    
    rng = np.random.default_rng(42)
    
    num_restarts = 30
    steps_per_restart = 10000
    
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        if r == 0:
            order = initial_order[:]
        elif r == 1:
            order = list(rng.permutation(n))
        elif r == 2:
            order = list(np.argsort(-degrees))
        elif r == 3:
            # Penalize vertices with high clique density among neighbors
            penalty = np.zeros(n)
            for i in range(n):
                nbrs = neighbors[i]
                if len(nbrs) > 1:
                    sub_adj = adjacency[np.ix_(nbrs, nbrs)]
                    penalty[i] = sub_adj.sum() / 2
            scores_init = -degrees - 0.3 * penalty
            order = list(np.argsort(scores_init))
        elif r == 4:
            scores_init = 1.0 / (1.0 + degrees.astype(float))
            order = list(np.argsort(-scores_init))
        else:
            noise_scale = max(1.0, degrees.std() * 0.3)
            noisy_deg = degrees + rng.normal(0, noise_scale, n)
            order = list(np.argsort(noisy_deg))
            
        current_size = greedy_mis_size(order)
        
        if current_size > best_size:
            best_size = current_size
            best_order = order[:]
            
        # Hill Climbing with mixed swap and insertion moves
        for step in range(steps_per_restart):
            if time.time() > deadline - 20:
                break
                
            if rng.random() < 0.5:
                # Swap
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
                    order[i], order[j] = order[j], order[i]
            else:
                # Insertion
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i == j:
                    continue
                v = order.pop(i)
                order.insert(j, v)
                
                new_size = greedy_mis_size(order)
                if new_size >= current_size:
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_order = order[:]
                else:
                    # Revert insertion
                    order.pop(j)
                    order.insert(i, v)
                
    # Final polish phase
    if time.time() < deadline - 10:
        order = best_order[:]
        current_size = best_size
        for step in range(50000):
            if time.time() > deadline - 5:
                break
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            v = order.pop(i)
            order.insert(j, v)
            
            new_size = greedy_mis_size(order)
            if new_size > current_size:
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_order = order[:]
            else:
                order.pop(j)
                order.insert(i, v)
                
    # Convert to scores
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
