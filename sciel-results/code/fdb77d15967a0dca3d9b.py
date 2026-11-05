import numpy as np
import time

def priority(adjacency):
    """
    Calculate priority scores for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Initialize scores using a "survival probability" heuristic:
       score ~ 1 / (degree + 1 + beta * avg_neighbor_degree)
       This models vertices that are structurally isolated from dense clusters
       as having higher survival probability in a random sequential elimination.
       (Adopted from Inspiration 1, which scored 1.1429)
       
    2. Refine these scores using a hybrid local search:
       - Rank-adjacent swaps (fine-grained exploitation)
       - Random swaps (global exploration)
       - Multi-restart with noise injection to escape local optima
       (Adapted from Current Program's machinery, extended time budget)
       
    3. Fast simulation using precomputed neighbor lists and vectorized deactivation.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
    
    # Ensure symmetry and remove self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    
    # Convert to boolean for faster neighbor lookups
    A_bool = A > 0.5
    
    # Precompute neighbor lists as arrays for fast simulation
    neighbors_list = [np.where(A_bool[i])[0] for i in range(N)]
    
    # --- 1. Initial Heuristic: Survival Probability (from Inspiration 1) ---
    deg = A.sum(axis=1)
    
    # Sum of degrees of neighbors
    neighbor_deg_sum = A.dot(deg)
    
    # Average degree of neighbors
    safe_deg = deg + 1e-6
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # Heuristic: Score = 1 / (degree + 1 + beta * avg_neighbor_degree)
    # Low degree and low local density -> high score -> picked first
    beta = 0.5
    raw_scores = 1.0 / (deg + 1.0 + beta * avg_neighbor_deg)
    
    # Normalize to [0, 1] for consistent noise scaling
    if raw_scores.max() > raw_scores.min():
        scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        scores = np.ones(N) / N
        
    def simulate_fast(s):
        """
        Fast greedy simulation.
        Returns the size of the independent set.
        """
        order = np.argsort(-s, kind='stable')
        active = np.ones(N, dtype=bool)
        count = 0
        
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count
    
    # --- 2. Baseline ---
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    # --- 3. Local Search with Multi-Restart ---
    rng = np.random.default_rng(42)
    
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Perturb the best score found so far
            # Noise scale increases slightly for later restarts to explore more
            noise_scale = 0.05 + 0.02 * restart
            noise = rng.normal(0, noise_scale, N)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)
            
            # If the perturbed score is significantly worse, it might be in a bad basin.
            # But let's give the local search a chance.
        
        # Local search loop
        no_improve = 0
        max_no_improve = 3000 + 500 * restart
        
        # Pre-compute current order for rank-adjacent swaps
        current_order = np.argsort(-current_scores, kind='stable')
        
        while time.time() - start_time < DEADLINE and no_improve < max_no_improve:
            # Choose swap candidates
            # 70% rank-adjacent (fine-grained), 30% random (exploration)
            if rng.random() < 0.7:
                # Pick a random position in the current order
                base_idx = rng.integers(0, N - 1)
                # Pick a nearby position (within 15 ranks)
                offset = rng.integers(1, min(16, N - 1))
                pos1 = base_idx
                pos2 = min(base_idx + offset, N - 1)
                i = current_order[pos1]
                j = current_order[pos2]
            else:
                i, j = rng.integers(0, N, size=2)
            
            if i == j:
                continue
            
            # Swap scores
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_val = simulate_fast(current_scores)
            
            if new_val > current_val:
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
                    # Update the order for rank-adjacent swaps
                    current_order = np.argsort(-current_scores, kind='stable')
            else:
                # Revert swap
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                no_improve += 1
        
        # Early stopping if no improvement in this restart and we've done several
        if current_val <= best_val and restart > 3:
            # Check if we're stuck overall
            if restart > 6:
                break
    
    return best_scores
