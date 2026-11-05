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
    
    # 1. Initial Heuristic: Enhanced Survival Probability Model
    # Use multiple structural features for better initialization
    deg = adj.sum(axis=1).astype(float)
    neighbor_deg_sum = adj @ deg
    safe_deg = deg + 1.0
    cluster_density = neighbor_deg_sum / safe_deg
    
    # Additional features: second-neighbor density and isolation score
    # Vertices with fewer neighbors that have few neighbors are more "isolated"
    second_deg = adj @ (adj @ deg)
    safe_second = second_deg + 1.0
    second_density = second_deg / safe_second
    
    # Combine: penalize degree, local density, and global connectivity
    scores = -deg - 0.3 * cluster_density - 0.1 * second_density
    
    # Normalize scores to [0, 1] range for better numerical behavior
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    def simulate_fast(s):
        """Fast greedy simulation with optimized neighbor updates."""
        order = np.argsort(-s, kind='stable')
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    # Only deactivate neighbors that are still active
                    for nb in nbrs:
                        if active[nb]:
                            active[nb] = False
        return count
    
    # 2. Baseline
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    # 3. Local Search with time guard and multiple restarts
    max_time = 115.0  # Safety margin before 120s limit
    rng = np.random.default_rng(42)
    
    # Precompute a list of promising swap candidates based on initial scores
    # Focus swaps on vertices with similar scores (small perturbations)
    score_order = np.argsort(-scores)
    
    # Try multiple restarts with different initial perturbations
    for restart in range(8):
        if time.time() - start_time > max_time - 5:
            break
        
        # For restart 0, use the original scores; for others, add noise
        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Perturb from best found so far with smaller noise
            noise_scale = 0.05 + 0.01 * restart
            noise = rng.normal(0, noise_scale, n)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)
        
        # Local search: try targeted pair swaps
        no_improve = 0
        max_no_improve = 1500 + 200 * restart  # Allow more iterations in later restarts
        
        # Track recent swaps to avoid cycling
        recent_swaps = set()
        
        while time.time() - start_time < max_time - 2 and no_improve < max_no_improve:
            # Choose swap candidates: focus on adjacent vertices in score ordering
            # or random pairs with some structure
            if rng.random() < 0.6:
                # Structured swap: pick vertices close in score rank
                base_idx = rng.integers(0, n - 1)
                offset = rng.integers(1, min(10, n - 1))
                i = score_order[base_idx]
                j = score_order[min(base_idx + offset, n - 1)]
            else:
                # Random swap
                i, j = rng.integers(0, n, size=2)
            
            if i == j:
                continue
            
            # Avoid recently tried swaps
            swap_key = (min(i, j), max(i, j))
            if swap_key in recent_swaps:
                continue
            recent_swaps.add(swap_key)
            if len(recent_swaps) > 500:
                recent_swaps.clear()
            
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
        if current_val <= best_val and restart > 2:
            break
    
    return best_scores
