"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

# Cache for pre-computed optimal depths
# Key: (sigma, A, k, gamma, Q, steps)
# Value: dict with 'bid' and 'ask' arrays of shape (steps, 2*Q+1)
_OPT_TABLES = {}

def _get_optimal_table(params):
    """
    Pre-compute optimal bid and ask depths for all (step, inventory) states
    using backward induction on the discrete HJB equation.
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["Q"], params["steps"])
    if key in _OPT_TABLES:
        return _OPT_TABLES[key]
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    
    # State space: n in [0, steps-1], q in [-Q, Q]
    # We compute V[n, q] which is C_n(q) from the decomposition V = m*q + C
    # V[steps, q] = 0 (no terminal charge)
    
    # Initialize value function for the last step (n = steps-1 uses V[steps])
    # V_next is an array of shape (2*Q+1,)
    V_next = np.zeros(2*Q + 1, dtype=np.float64)
    
    # Store optimal depths
    opt_bid = np.zeros((steps, 2*Q + 1), dtype=np.float64)
    opt_ask = np.zeros((steps, 2*Q + 1), dtype=np.float64)
    
    # Use a finer depth grid for better accuracy
    max_depth = 8.0 / k
    depth_grid = np.linspace(0.0, max_depth, 100)
    
    # Precompute probabilities for the depth grid
    # p(b) = 1 - exp(-A * exp(-k*b) * dt)
    p_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(2*Q + 1, dtype=np.float64)
        
        for i, q in enumerate(range(-Q, Q + 1)):
            # Inventory cost
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Calculate Deltas
            # Handle boundaries
            if q < Q:
                delta_b = V_next[q + 1 + Q] - V_next[q + Q]
            else:
                delta_b = 0.0 # Will be forced to b=inf, so term is 0
            
            if q > -Q:
                delta_a = V_next[q - 1 + Q] - V_next[q + Q]
            else:
                delta_a = 0.0 # Will be forced to a=inf, so term is 0
            
            # Find optimal b
            if q < Q:
                # f(b) = p_b * (b + delta_b)
                # We want to maximize this over the depth grid
                # Also consider b=inf (no quote) which gives 0
                vals_b = p_grid * (depth_grid + delta_b)
                max_f = np.max(vals_b)
                # If max_f < 0, prefer no quote (b=inf)
                if max_f < 0:
                    best_b = np.inf
                    contrib_b = 0.0
                else:
                    best_b = depth_grid[np.argmax(vals_b)]
                    contrib_b = max_f
            else:
                best_b = np.inf
                contrib_b = 0.0
            
            # Find optimal a
            if q > -Q:
                # g(a) = p_a * (a + delta_a)
                vals_a = p_grid * (depth_grid + delta_a)
                max_g = np.max(vals_a)
                # If max_g < 0, prefer no quote (a=inf)
                if max_g < 0:
                    best_a = np.inf
                    contrib_a = 0.0
                else:
                    best_a = depth_grid[np.argmax(vals_a)]
                    contrib_a = max_g
            else:
                best_a = np.inf
                contrib_a = 0.0
            
            V_curr[i] = -cost + V_next[q + Q] + contrib_b + contrib_a
            
            opt_bid[n, i] = best_b
            opt_ask[n, i] = best_a
        
        V_next = V_curr
    
    table = {'bid': opt_bid, 'ask': opt_ask}
    _OPT_TABLES[key] = table
    return table


def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed HJB solution.
    """
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Fallback to simple safe heuristic if running out of time
        sigma = params["sigma"]
        k = params["k"]
        Q = params["Q"]
        q = np.asarray(inventory, dtype=np.float64)
        base_depth = 1.0 / k
        bid_depth = np.full_like(q, base_depth)
        ask_depth = np.full_like(q, base_depth)
        bid_depth[q >= Q] = np.inf
        ask_depth[q <= -Q] = np.inf
        return bid_depth, ask_depth

    # Get or compute table
    table = _get_optimal_table(params)
    
    # Map t to step index
    steps = params["steps"]
    T = params["T"]
    dt = T / steps
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
    
    # Map inventory to index
    Q = params["Q"]
    q = np.asarray(inventory, dtype=np.int64)
    # Clamp to [-Q, Q] just in case, though contract says it's in range
    q_clamped = np.clip(q, -Q, Q)
    indices = q_clamped + Q
    
    bid_depths = table['bid'][n, indices]
    ask_depths = table['ask'][n, indices]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
