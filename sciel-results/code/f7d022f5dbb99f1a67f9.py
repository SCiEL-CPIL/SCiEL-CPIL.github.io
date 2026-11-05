import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Survival Probability Scoring + Optimized Local Search
    1. Initialize scores using a dynamic survival probability model.
    2. Use fast greedy simulation with precomputed neighbor sets.
    3. Local search with boundary-aware swap candidates.
    4. Multiple restarts with adaptive noise.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists as sorted arrays for fast lookup
    neighbors_list = [np.where(adj[i])[0] for i in range(n)]
    
    # 1. Initial Heuristic: Survival Probability Model
    # Vertices with fewer neighbors in dense clusters have higher survival probability
    deg = adj.sum(axis=1).astype(float)
    neighbor_deg_sum = adj @ deg
    safe_deg = deg + 1.0
    cluster_density = neighbor_deg_sum / safe_deg
    
    # Second-neighbor density (captures structural isolation)
    second_deg = adj @ (adj @ deg)
    safe_second = second_deg + 1.0
    second_density = second_deg / safe_second
    
    # Survival probability: lower degree and lower cluster density = higher survival
    # Use inverse to emphasize structural isolation
    survival_prob = 1.0 / (deg + 1.0) - 0.3 * cluster_density / (deg + 1.0) - 0.1 * second_density / (deg + 1.0)
    scores = survival_prob
    
    # Normalize to [0, 1]
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    def simulate_fast(s):
        """Fast greedy simulation using precomputed neighbors."""
        order = np.argsort(-s, kind='stable')
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count
    
    # 2. Baseline
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    # 3. Local Search with time guard
    max_time = 118.0  # Safety margin before 150s limit
    rng = np.random.default_rng(42)
    
    # Precompute score ordering
    score_order = np.argsort(-scores)
    
    # Try multiple restarts
    for restart in range(15):
        if time.time() - start_time > max_time - 3:
            break
        
        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Adaptive noise: start larger, decrease
            noise_scale = max(0.01, 0.06 - 0.004 * restart)
            noise = rng.normal(0, noise_scale, n)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)
        
        no_improve = 0
        max_no_improve = 3000 + 500 * restart
        
        recent_swaps = set()
        
        while time.time() - start_time < max_time - 1 and no_improve < max_no_improve:
            # Smart swap: focus on vertices near the selection boundary
            if rng.random() < 0.6:
                # Pick a random position and swap with a nearby vertex
                base_idx = rng.integers(0, n - 1)
                offset = rng.integers(1, min(20, n - 1))
                i = score_order[base_idx]
                j = score_order[min(base_idx + offset, n - 1)]
            elif rng.random() < 0.5:
                # Swap with a vertex that has similar degree (likely similar impact)
                i = rng.integers(0, n)
                # Find a vertex with similar degree
                target_deg = deg[i]
                candidates = np.where(np.abs(deg - target_deg) < 2)[0]
                if len(candidates) > 0:
                    j = candidates[rng.integers(0, len(candidates))]
                else:
                    j = rng.integers(0, n)
            else:
                # Random swap
                i, j = rng.integers(0, n, size=2)
            
            if i == j:
                continue
            
            swap_key = (min(i, j), max(i, j))
            if swap_key in recent_swaps:
                continue
            recent_swaps.add(swap_key)
            if len(recent_swaps) > 2000:
                recent_swaps.clear()
            
            # Swap scores
            si, sj = current_scores[i], current_scores[j]
            current_scores[i], current_scores[j] = sj, si
            new_val = simulate_fast(current_scores)
            
            if new_val > current_val:
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
            else:
                current_scores[i], current_scores[j] = si, sj
                no_improve += 1
        
        # Early exit if no progress
        if current_val <= best_val and restart > 5:
            break
    
    return best_scores
