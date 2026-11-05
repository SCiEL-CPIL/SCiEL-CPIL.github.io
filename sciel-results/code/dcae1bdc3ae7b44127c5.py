import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Hub-Blocking Heuristic + Local Search with Polish Pass
    1. Initialize scores using hub-blocking: prioritize low-degree vertices adjacent
       to high-degree vertices (prune hubs without selecting them).
    2. Use local search (pair swaps) to directly optimize the greedy decoder output.
    3. Final polish pass: targeted single-vertex score adjustments for fine-tuning.
    4. Multiple restarts with different random seeds to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast simulation
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    # 1. Initial Heuristic: Hub-Blocking Score
    # Prioritize low-degree vertices that are adjacent to high-degree vertices
    deg = adj.sum(axis=1).astype(float)
    
    # For each vertex, compute the average degree of its neighbors
    neighbor_deg_sum = adj @ deg
    safe_deg = deg + 1.0
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # Hub-blocking score: low degree + high neighbor degree = good priority
    # We want to select vertices that are "leaves" attached to hubs
    hub_blocking_score = -deg + 0.5 * avg_neighbor_deg
    
    # Also incorporate local density penalty
    second_deg = adj @ (adj @ deg)
    safe_second = second_deg + 1.0
    second_density = second_deg / safe_second
    
    # Combine: hub-blocking dominates, with some penalty for dense local structure
    scores = hub_blocking_score - 0.1 * second_density
    
    # Normalize scores to [0, 1] range
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
                    for nb in nbrs:
                        if active[nb]:
                            active[nb] = False
        return count
    
    # 2. Baseline
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    # 3. Local Search with time guard and multiple restarts
    rng = np.random.default_rng(42)
    
    # Precompute score order for structured swaps
    score_order = np.argsort(-scores)
    
    # Try multiple restarts with different initial perturbations
    for restart in range(8):
        if time.time() > deadline - 10:
            break
        
        # For restart 0, use the original scores; for others, add noise
        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Perturb from best found so far with smaller noise
            noise_scale = 0.03 + 0.01 * restart
            noise = rng.normal(0, noise_scale, n)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)
        
        # Local search: try targeted pair swaps
        no_improve = 0
        max_no_improve = 1000 + 150 * restart
        
        # Track recent swaps to avoid cycling
        recent_swaps = set()
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            # Choose swap candidates
            if rng.random() < 0.6:
                # Structured swap: pick vertices close in score rank
                base_idx = rng.integers(0, n - 1)
                offset = rng.integers(1, min(15, n - 1))
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
            if len(recent_swaps) > 800:
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
    
    # 4. Polish Pass: Fine-tune with single-vertex score adjustments
    if time.time() < deadline - 3:
        polish_scores = best_scores.copy()
        polish_val = best_val
        polish_rng = np.random.default_rng(123)
        
        # Try small score perturbations on individual vertices
        for _ in range(2000):
            if time.time() > deadline - 2:
                break
            
            v = polish_rng.integers(0, n)
            # Small perturbation
            delta = polish_rng.normal(0, 0.02)
            original = polish_scores[v]
            polish_scores[v] = original + delta
            
            new_val = simulate_fast(polish_scores)
            
            if new_val > polish_val:
                polish_val = new_val
                if polish_val > best_val:
                    best_val = polish_val
                    best_scores = polish_scores.copy()
            else:
                polish_scores[v] = original
    
    return best_scores
