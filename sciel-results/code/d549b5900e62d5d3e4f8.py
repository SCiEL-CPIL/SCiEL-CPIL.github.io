import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Maximum Entropy Scoring + Local Search Optimization
    1. Initialize scores using a "survival probability" model: vertices that are
       structurally isolated from dense clusters get higher priority.
    2. Use local search (pair swaps) to directly optimize the greedy decoder output.
    3. Multiple restarts with different random seeds to escape local optima.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast simulation
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    # 1. Initial Heuristic: Survival Probability Model
    # The idea: a vertex's "survival probability" in a random sequential elimination
    # process is approximated by how isolated it is from dense clusters.
    # We use: score = -deg - alpha * (neighbor degree sum) / (deg + 1)
    # This penalizes vertices adjacent to high-degree hubs (dense clusters)
    deg = adj.sum(axis=1).astype(float)
    neighbor_deg_sum = adj @ deg
    safe_deg = deg + 1.0
    # Survival probability proxy: lower neighbor density means more isolated
    cluster_density = neighbor_deg_sum / safe_deg
    scores = -deg - 0.3 * cluster_density
    
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
