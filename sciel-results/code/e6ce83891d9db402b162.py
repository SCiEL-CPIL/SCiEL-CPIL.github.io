import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Parallel Discrete-Time Exclusion Process + Local Search
    1. Simulate a parallel exclusion process where vertices with fewest active
       neighbors are selected first in each round.
    2. Assign priority based on earliest selection time (dynamic interference).
    3. Use local search (pair swaps) to directly optimize the greedy decoder output.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast simulation
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    # 1. Initial Heuristic: Parallel Discrete-Time Exclusion Process
    # Simulate rounds: in each round, select vertices with fewest active neighbors
    # that haven't been selected or eliminated yet.
    # Priority = -selection_time (earlier selection = higher priority)
    
    active = np.ones(n, dtype=bool)
    selected_time = np.full(n, n + 1, dtype=float)  # default: never selected
    round_num = 0
    
    while active.any():
        # Count active neighbors for each active vertex
        active_nbr_count = np.zeros(n, dtype=int)
        for v in np.where(active)[0]:
            nbrs = neighbors_list[v]
            if nbrs:
                active_nbr_count[v] = np.sum(active[nbrs])
        
        # Find vertices with minimum active neighbor count among active vertices
        min_count = np.min(active_nbr_count[active])
        # Get all vertices with this minimum count
        candidates = np.where(active & (active_nbr_count == min_count))[0]
        
        if len(candidates) == 0:
            break
            
        # Select all candidates in this round
        for v in candidates:
            selected_time[v] = round_num
            active[v] = False
            # Eliminate neighbors
            nbrs = neighbors_list[v]
            if nbrs:
                active[nbrs] = False
        
        round_num += 1
        if time.time() - start_time > 110:
            break
    
    # Vertices never selected get high time (low priority)
    # Score: earlier selection = higher priority
    scores = -selected_time
    
    # Add small tie-breaker based on degree to break symmetries
    deg = adj.sum(axis=1).astype(float)
    scores = scores - 0.01 * deg
    
    def simulate_fast(s):
        """Fast greedy simulation using numpy boolean array."""
        order = np.argsort(-s, kind='stable')
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        return count
    
    # 2. Baseline
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    # 3. Local Search with time guard and multiple restarts
    max_time = 115.0  # Safety margin before 120s limit
    rng = np.random.default_rng(42)
    
    # Try multiple restarts with different initial perturbations
    for restart in range(5):
        if time.time() - start_time > max_time - 5:
            break
        
        # For restart 0, use the original scores; for others, add noise
        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Perturb from best found so far
            noise = rng.normal(0, 0.1, n)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)
        
        # Local search: try pair swaps
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() - start_time < max_time - 2 and no_improve < max_no_improve:
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            
            # Swap
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_val = simulate_fast(current_scores)
            
            if new_val > current_val:
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
            else:
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                no_improve += 1
        
        # If no improvement after this restart, break early
        if current_val <= best_val and restart > 0:
            break
    
    return best_scores
