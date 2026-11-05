import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Mean-Field Game / Reaction-Diffusion Process
    1. Model vertex priorities as agents in a continuous-time game.
    2. Update scores via a reaction-diffusion PDE:
       ds/dt = D * L * s + R(s, deg)
       where L is the graph Laplacian (diffusion) and R is a reaction term
       that favors low-degree nodes (source) and penalizes high-degree nodes.
    3. Run the simulation until convergence or time budget is exhausted.
    4. The resulting steady-state scores are used for the greedy decoder.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast greedy simulation
    neighbors_list = [np.where(adj[i])[0] for i in range(n)]
    
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

    # --- Mean-Field Game Setup ---
    # Degrees
    deg = adj.sum(axis=1).astype(float)
    
    # Graph Laplacian: L = D - A
    # For diffusion, we use the normalized Laplacian for stability
    deg_safe = deg + 1e-6
    D_inv = 1.0 / deg_safe
    # Normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    # But for reaction-diffusion, standard Laplacian L = D - A is often used
    # Let's use the standard Laplacian for diffusion of "priority mass"
    # ds_i/dt = sum_j (s_j - s_i) * w_ij  (diffusion)
    # This is equivalent to L * s where L = D - A
    
    # Precompute L * s operation efficiently
    # L = D - A
    # L @ s = deg * s - A @ s
    
    # Reaction term R:
    # We want low-degree nodes to have higher priority.
    # Let R_i = alpha / (deg_i + 1) - beta * s_i
    # This creates a source for low-degree nodes and a sink proportional to current score
    # to prevent unbounded growth and encourage equilibrium.
    
    alpha = 1.0
    beta = 0.1
    
    # Diffusion coefficient
    D_coeff = 0.5
    
    # Initial scores: random or based on degree
    # Start with a bias towards low degree
    s = 1.0 / (deg + 1.0)
    
    # Normalize initial scores
    s = s / s.sum()
    
    # Simulation parameters
    dt = 0.05
    max_steps = 20000
    time_limit = 118.0  # Safety margin
    
    # Adaptive step size: start small, increase if stable
    current_dt = dt
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Compute Laplacian term: L @ s = deg * s - A @ s
        # A @ s can be computed as adj @ s
        As = adj @ s
        Ls = deg * s - As
        
        # Compute Reaction term: R = alpha / (deg + 1) - beta * s
        R = alpha / (deg + 1.0) - beta * s
        
        # Update: s_new = s + dt * (D_coeff * Ls + R)
        ds = D_coeff * Ls + R
        s_new = s + current_dt * ds
        
        # Ensure non-negativity (scores should be positive for ranking)
        s_new = np.maximum(s_new, 0.0)
        
        # Normalize to prevent numerical drift
        s_new = s_new / (s_new.sum() + 1e-12)
        
        # Check for convergence
        diff = np.max(np.abs(s_new - s))
        s = s_new
        
        if diff < 1e-6:
            break
        
        # Adaptive dt: increase if stable, decrease if oscillating
        if step % 100 == 0:
            if diff < 1e-4:
                current_dt = min(current_dt * 1.2, 0.2)
            else:
                current_dt = max(current_dt * 0.8, 0.01)
    
    # Final scores
    scores = s
    
    # Normalize to [0, 1] for consistency
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    # Optional: Local search refinement if time permits
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0:
        # Simple hill-climbing to polish the mean-field solution
        rng = np.random.default_rng(42)
        score_order = np.argsort(-scores)
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() - start_time < time_limit - 1 and no_improve < max_no_improve:
            # Random swap
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            
            si, sj = best_scores[i], best_scores[j]
            best_scores[i], best_scores[j] = sj, si
            new_val = simulate_fast(best_scores)
            
            if new_val > best_val:
                best_val = new_val
                no_improve = 0
            else:
                best_scores[i], best_scores[j] = si, sj
                no_improve += 1
    
    return best_scores
