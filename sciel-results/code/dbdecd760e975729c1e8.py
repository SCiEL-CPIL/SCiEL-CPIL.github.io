import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for Python overhead/returns
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Pre-compute neighbor indices for each vertex for fast blocking
    # neighbors[i] is an array of indices of vertices connected to i
    neighbors = [np.where(adj[i])[0] for i in range(n)]
    # Also include self? No, the loop handles `blocked[v]` explicitly.
    # But wait, if we select v, we block v and neighbors.
    
    # Initial heuristic: Least Degree First
    deg = adj.sum(axis=1)
    initial_order = np.argsort(deg)
    
    best_order = initial_order.copy()
    
    def get_size(order):
        """
        Fast greedy MIS size calculation.
        """
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not blocked[v]:
                count += 1
                blocked[v] = True
                # Block neighbors
                if len(neighbors[v]) > 0:
                    blocked[neighbors[v]] = True
        return count

    best_size = get_size(best_order)
    current_order = best_order.copy()
    current_size = best_size
    
    rng = np.random.default_rng()
    
    temp = 1.0
    cooling = 0.9998
    min_temp = 1e-3
    
    # Run optimization
    while time.time() - start_time < time_limit:
        # Random move: swap two positions in the ordering
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        # Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_size = get_size(current_order)
        delta = new_size - current_size
        
        if delta > 0:
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_order = current_order.copy()
        else:
            # Accept or reject
            if temp > min_temp:
                # Probability of accepting worse move
                # delta is negative
                prob = np.exp(delta / temp)
                if rng.random() < prob:
                    current_size = new_size
                else:
                    # Reject
                    current_order[i], current_order[j] = current_order[j], current_order[i]
            else:
                # Hill climbing mode: reject worse
                current_order[i], current_order[j] = current_order[j], current_order[i]
                
        temp *= cooling
        
        # If we've been stuck for a long time (optional), we could restart, 
        # but simple SA usually works well enough for N=200.
        
    # Map best_order to scores
    # best_order is a permutation of vertices [v_0, v_1, ..., v_{n-1}]
    # We want v_0 to have highest score.
    # score[v_k] = n - 1 - k
    scores = np.zeros(n)
    scores[best_order] = np.arange(n - 1, -1, -1)
    
    return scores
