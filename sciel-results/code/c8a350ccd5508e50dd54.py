import numpy as np
from functools import lru_cache

# Global cache for DP solutions
_DP_CACHE = {}

def _solve_scalar(d, delta_v, K, k):
    """Solve K(d + delta_v)exp(-kd) = 1 for d >= 0 using Newton's method."""
    # If delta_v is very large, d is close to 0.
    # If delta_v is very negative, d is large.
    # g(d) = ln(K(d+delta_v)) - kd - 1
    # g'(d) = 1/(d+delta_v) - k
    
    # Initial guess
    # Approximation: d approx 1/k - delta_v
    d_guess = max(0.0, 1.0/k - delta_v)
    
    for _ in range(20):
        val = K * (d_guess + delta_v) * np.exp(-k * d_guess)
        if val < 1e-12:
            # d is too large or delta_v too negative
            d_guess += 1.0
            continue
            
        # g(d) = ln(val) - 1
        g = np.log(val) - 1.0
        # g'(d) = 1/(d+delta_v) - k
        denom = d_guess + delta_v
        if denom < 1e-9:
             # Singular, move d
             d_guess += 0.1
             continue
        gp = 1.0/denom - k
        
        if abs(gp) < 1e-12:
            break
            
        step = g / gp
        d_new = d_guess - step
        
        if d_new < 0:
            d_new = 0.0
            if d_guess < 1e-6:
                break
            d_guess = d_new
        else:
            if abs(d_new - d_guess) < 1e-8:
                d_guess = d_new
                break
            d_guess = d_new
            
    return max(0.0, d_guess)

def _compute_dp(params):
    """Compute optimal depths for all steps and inventories."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    K = A * k * dt
    
    num_q = 2 * Q + 1
    # V_next[i] corresponds to q = i - Q
    V_next = np.zeros(num_q)
    
    # Lookup tables
    bid_depths = np.full((steps, num_q), np.inf)
    ask_depths = np.full((steps, num_q), np.inf)
    
    # Penalty coefficient
    penalty_coeff = 0.5 * gamma * (sigma ** 2) * dt
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(num_q)
        for i in range(num_q):
            q = i - Q
            
            # Calculate deltas
            # Bid fill: q -> q+1
            if q < Q:
                delta_b = V_next[i+1] - V_next[i]
            else:
                delta_b = -np.inf # No bid fill allowed
            
            # Ask fill: q -> q-1
            if q > -Q:
                delta_a = V_next[i-1] - V_next[i]
            else:
                delta_a = -np.inf # No ask fill allowed
            
            d_b = 0.0
            d_a = 0.0
            p_b = 0.0
            p_a = 0.0
            
            if delta_b > -np.inf:
                d_b = _solve_scalar(0.0, delta_b, K, k)
                bid_depths[n, i] = d_b
                p_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
            
            if delta_a > -np.inf:
                d_a = _solve_scalar(0.0, delta_a, K, k)
                ask_depths[n, i] = d_a
                p_a = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)
            
            # Expected Value
            val = 0.0
            # Reward
            val += p_b * d_b
            val += p_a * d_a
            
            # Next Value
            # Case 1: No fill
            val += (1 - p_b) * (1 - p_a) * V_next[i]
            # Case 2: Bid fill only
            if q < Q:
                val += p_b * (1 - p_a) * V_next[i+1]
            # Case 3: Ask fill only
            if q > -Q:
                val += p_a * (1 - p_b) * V_next[i-1]
            # Case 4: Both fill
            if q < Q and q > -Q:
                val += p_b * p_a * V_next[i] # q+1-1 = q
            
            # Penalty
            val -= penalty_coeff * q * q
            
            V_curr[i] = val
        
        V_next = V_curr
        
    return bid_depths, ask_depths

def quote(t, inventory, mid, params):
    """
    Optimal market making with inventory risk using DP on inventory grid.
    """
    # Cache key
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    
    if key not in _DP_CACHE:
        _DP_CACHE[key] = _compute_dp(params)
    
    bid_depths_table, ask_depths_table = _DP_CACHE[key]
    
    Q = params["Q"]
    steps = params["steps"]
    
    # Determine step index
    # t = n * dt, dt = T / steps
    dt = params["T"] / steps
    n = int(round(t / dt))
    n = np.clip(n, 0, steps - 1)
    
    # Map inventory to index
    # inventory is array of ints in [-Q, Q]
    inv_indices = inventory + Q
    
    # Lookup
    bid_depths = bid_depths_table[n, inv_indices]
    ask_depths = ask_depths_table[n, inv_indices]
    
    return bid_depths, ask_depths
