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

def _solve_optimal_depths_vectorized(A, k, dt, deltas):
    """
    Solve max_{x >= 0} p(x) * (x + delta) where p(x) = 1 - exp(-A exp(-kx) dt).
    Uses first-order approximation as initial guess, then Newton refinement.
    """
    n = len(deltas)
    best_x = np.full(n, np.inf, dtype=np.float64)
    best_val = np.zeros(n, dtype=np.float64)
    
    # Parameters
    lam = A * dt
    if lam <= 0:
        return best_x, best_val
        
    # First-order approximation: x* = (1/k) * (1 - delta)
    x_approx = (1.0 / k) * (1.0 - deltas)
    
    # Only consider positive depths
    valid_mask = x_approx > 0
    if not np.any(valid_mask):
        return best_x, best_val
    
    idx = np.where(valid_mask)[0]
    x = x_approx[idx].copy()
    d = deltas[idx]
    
    # Newton iterations to refine
    # f(x) = p(x)*(x+d), p(x) = 1-exp(-lam*exp(-kx))
    # f'(x) = p'(x)*(x+d) + p(x)
    # p'(x) = -lam*k*exp(-kx)*exp(-lam*exp(-kx)) ... wait
    # p(x) = 1 - exp(-lam*exp(-kx))
    # p'(x) = -exp(-lam*exp(-kx)) * lam*k*exp(-kx) = -lam*k*exp(-kx)*exp(-lam*exp(-kx))
    # f'(x) = p'(x)*(x+d) + p(x)
    # f''(x) = p''(x)*(x+d) + 2*p'(x)
    # p''(x) = -lam*k*exp(-kx)*exp(-lam*exp(-kx)) * (-k) + (-lam*k*exp(-kx))*(-lam*k*exp(-kx)*exp(-lam*exp(-kx)))
    #        = lam*k^2*exp(-2kx)*exp(-lam*exp(-kx)) + lam^2*k^2*exp(-2kx)*exp(-2*lam*exp(-kx))
    
    for _newton_iter in range(4):
        # Compute exp(-kx)
        ex = np.exp(-k * x)
        # Compute lam*exp(-kx)
        lex = lam * ex
        # Clip to avoid overflow
        lex_clipped = np.clip(lex, -50, 50)
        # p(x) = 1 - exp(-lex)
        p = 1.0 - np.exp(-lex_clipped)
        # p'(x) = -lam*k*exp(-kx)*exp(-lex)
        p_prime = -lam * k * ex * np.exp(-lex_clipped)
        # f(x) = p*(x+d)
        f = p * (x + d)
        # f'(x) = p'(x)*(x+d) + p(x)
        fp = p_prime * (x + d) + p
        # f''(x) = p''(x)*(x+d) + 2*p'(x)
        # p''(x) = lam*k^2*exp(-2kx)*exp(-lex) + lam^2*k^2*exp(-2kx)*exp(-2*lex)
        p_dd = lam * k * k * ex * ex * np.exp(-lex_clipped) + \
               lam * lam * k * k * ex * ex * np.exp(-2.0 * lex_clipped)
        fpp = p_dd * (x + d) + 2.0 * p_prime
        
        # Newton step
        mask_nonzero = np.abs(fpp) > 1e-30
        dx = np.zeros_like(x)
        dx[mask_nonzero] = -f[mask_nonzero] / fpp[mask_nonzero]
        # Limit step size
        dx = np.clip(dx, -0.5 * k, 0.5 * k)
        x = x + dx
        # Ensure x > 0
        x = np.maximum(x, 1e-10)
    
    # Final evaluation
    ex = np.exp(-k * x)
    lex_clipped = np.clip(lam * ex, -50, 50)
    p = 1.0 - np.exp(-lex_clipped)
    val = p * (x + d)
    
    # Check if quoting is beneficial (val > 0)
    quote_mask = val > 0
    
    best_x[idx] = np.where(quote_mask, x, np.inf)
    best_val[idx] = np.where(quote_mask, val, 0.0)
    
    return best_x, best_val

def _get_optimal_table(params):
    """
    Pre-compute optimal bid and ask depths for all (step, inventory) states
    using backward induction on the discrete HJB equation.
    Uses fast approximation for depth calculation.
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
    
    # Initialize value function
    V_next = np.zeros(2*Q + 1, dtype=np.float64)
    
    # Store optimal depths
    opt_bid = np.zeros((steps, 2*Q + 1), dtype=np.float64)
    opt_ask = np.zeros((steps, 2*Q + 1), dtype=np.float64)
    
    # Pre-compute costs
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 10.0:
            break
            
        V_curr = np.zeros(2*Q + 1, dtype=np.float64)
        
        delta_b = np.zeros(2*Q + 1, dtype=np.float64)
        delta_a = np.zeros(2*Q + 1, dtype=np.float64)
        
        # Valid bids: q < Q => indices 0 to 2Q-1
        delta_b[:2*Q] = V_next[1:2*Q+1] - V_next[:2*Q]
        
        # Valid asks: q > -Q => indices 1 to 2Q
        delta_a[1:] = V_next[:2*Q] - V_next[1:]
        
        # Solve for optimal depths (fast approximation)
        bid_depths, bid_vals = _solve_optimal_depths_vectorized(A, k, dt, delta_b)
        ask_depths, ask_vals = _solve_optimal_depths_vectorized(A, k, dt, delta_a)
        
        # Mask out invalid quotes
        bid_depths[2*Q] = np.inf
        bid_vals[2*Q] = 0.0
        
        ask_depths[0] = np.inf
        ask_vals[0] = 0.0
        
        # V_curr = -cost + V_next + contrib_b + contrib_a
        V_curr = -cost + V_next + bid_vals + ask_vals
        
        opt_bid[n, :] = bid_depths
        opt_ask[n, :] = ask_depths
        
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
    if n < 0:
        n = 0
    
    # Map inventory to index
    Q = params["Q"]
    q = np.asarray(inventory, dtype=np.int64)
    # Clamp to [-Q, Q] just in case
    q_clamped = np.clip(q, -Q, Q)
    indices = q_clamped + Q
    
    bid_depths = table['bid'][n, indices]
    ask_depths = table['ask'][n, indices]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
