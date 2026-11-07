import numpy as np
import time

# Track the number of calls to estimate which round we're in
_call_count = 0

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step of guided local search with adaptive penalties.
    
    Uses a composite penalty:
    1. Cumulative penalty based on edge usage frequency (saturating, adaptive).
    2. Immediate penalty for edges in the current local optimum tour (adaptive).
    
    The penalty strength increases over rounds to escape deeper local optima.
    """
    global _call_count
    _call_count += 1
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Initialize output with true distances
    d_out = edge_distance.copy()
    
    if n == 0:
        return d_out
        
    # 1. Build mask for edges in the current local optimal tour
    tour_mask = np.zeros((n, n), dtype=float)
    
    try:
        for i in range(n):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % n]
            if 0 <= u < n and 0 <= v < n:
                tour_mask[u, v] = 1.0
                tour_mask[v, u] = 1.0
    except Exception:
        return d_out

    # 2. Estimate the current round (each round has 3 instances, so round = _call_count // 3)
    # But we don't know the exact mapping, so use a simpler approach:
    # Use the max value of edge_n_used to estimate how many rounds have passed
    max_used = np.max(edge_n_used) if edge_n_used.size > 0 else 0
    
    # 3. Calculate adaptive penalty parameters
    # As rounds progress, increase penalty strength to escape local optima
    # round_estimate based on max usage (each round adds 1 to used edges)
    round_est = max_used
    
    # Adaptive scaling: penalties grow with round number but saturate
    # Base penalty for cumulative usage
    alpha_cum_base = 0.3
    # Current tour edge penalty
    alpha_curr_base = 0.5
    
    # Increase with round but cap to avoid excessive perturbation
    growth_factor = 1.0 + 0.1 * min(round_est, 10)
    
    alpha_cum = alpha_cum_base * growth_factor
    alpha_curr = alpha_curr_base * growth_factor
    
    # Cap the penalties to prevent extreme values
    alpha_cum = min(alpha_cum, 2.0)
    alpha_curr = min(alpha_curr, 3.0)
    
    # 4. Calculate Cumulative Penalty using a smoother saturation
    # Use tanh for smoother behavior than linear saturation
    cum_penalty = np.tanh(edge_n_used / 3.0)
    
    # 5. Combine Penalties
    penalty_factor = 1.0 + alpha_cum * cum_penalty + alpha_curr * tour_mask
    
    # Apply penalty
    d_out *= penalty_factor
    
    # Safety check: ensure finite values
    if not np.all(np.isfinite(d_out)):
        return edge_distance.copy()
        
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 2.5:
        # Shouldn't happen for N=100, but just in case
        return edge_distance.copy()
        
    return d_out
