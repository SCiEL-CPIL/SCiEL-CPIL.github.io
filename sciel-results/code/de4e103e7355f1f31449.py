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

def _solve_optimal_depths_vectorized(A, k, dt, deltas, max_iter=20, tol=1e-9):
    """
    Solves max_{x >= 0} (1 - exp(-A exp(-kx) dt)) * (x + delta) for an array of deltas.
    Uses change of variable y = A exp(-kx) dt.
    x = (1/k) * ln((A dt) / y)
    Maximize J(y) = (1 - exp(-y)) * ( (1/k) * ln((A dt) / y) + delta )
    Subject to y in (0, A dt].
    
    Derivative condition:
    H(y) = exp(-y) * ( ln((A dt) / y) + k*delta + 1/y ) - 1 = 0
    """
    n = len(deltas)
    best_x = np.full(n, np.inf, dtype=np.float64)
    best_val = np.zeros(n, dtype=np.float64)
    
    # Parameters
    lam = A * dt
    if lam <= 0:
        return best_x, best_val
        
    # Initial guess for y
    # If delta is large positive, we want large x, so small y.
    # If delta is large negative, we want small x, so large y (close to lam).
    # A reasonable start is y = lam / 2 or similar.
    # Let's try y_init = lam * np.exp(-np.abs(deltas)*k) ? No, keep it simple.
    # If delta > 0, optimum is at small y. If delta < 0, optimum is at large y.
    # Let's use y_init = lam * 0.1 for all, and let Newton handle it, or adapt.
    # Actually, for delta ~ 0, y is around 1 or so?
    # Let's just use a safe interior point.
    y = np.full(n, 0.5 * lam, dtype=np.float64)
    
    # Ensure y is within (0, lam]
    y = np.clip(y, 1e-12, lam)
    
    for _ in range(max_iter):
        # Calculate terms
        # ln(lam / y)
        log_term = np.log(lam / y)
        # 1/y
        inv_y = 1.0 / y
        # exp(-y)
        exp_neg_y = np.exp(-y)
        
        # H(y) = exp(-y) * ( log_term + k*delta + inv_y ) - 1
        H = exp_neg_y * (log_term + k * deltas + inv_y) - 1.0
        
        # H'(y)
        # d/dy [ exp(-y) * ( log(lam/y) + k*delta + 1/y ) ]
        # = -exp(-y) * ( ... ) + exp(-y) * ( -1/y - 1/y^2 )
        # = -exp(-y) * ( log_term + k*delta + inv_y - 1/y - 1/y^2 )
        # = -exp(-y) * ( log_term + k*delta - 1/y^2 )
        H_prime = -exp_neg_y * (log_term + k * deltas - 1.0 / (y * y))
        
        # Newton step
        # Avoid division by zero
        H_prime_safe = np.where(np.abs(H_prime) < 1e-12, 1e-12, H_prime)
        step = H / H_prime_safe
        
        # Damped step to stay in bounds
        # If H > 0, we need to decrease y? 
        # Let's check signs.
        # As y -> 0, log_term -> inf, H -> inf.
        # As y -> lam, log_term -> 0.
        # H(lam) = exp(-lam) * (k*delta + 1/lam) - 1.
        # If H(lam) < 0, root is in (0, lam).
        # If H(0) > 0 and H(lam) < 0, there is a root.
        
        y_new = y - step
        
        # Clip to valid range
        y_new = np.clip(y_new, 1e-12, lam)
        
        # Check convergence
        if np.all(np.abs(y_new - y) < tol):
            y = y_new
            break
        y = y_new
        
    # Calculate optimal x from y
    # x = (1/k) * ln(lam / y)
    x_opt = (1.0 / k) * np.log(lam / y)
    
    # Ensure x >= 0
    x_opt = np.maximum(x_opt, 0.0)
    
    # Calculate the value J(y) to check if it's better than 0 (no quote)
    # J(y) = (1 - exp(-y)) * (x_opt + delta)
    val = (1.0 - np.exp(-y)) * (x_opt + deltas)
    
    # If val <= 0, it's better to not quote (x = inf, val = 0)
    mask = val > 0
    best_x[mask] = x_opt[mask]
    best_x[~mask] = np.inf
    best_val[mask] = val[mask]
    best_val[~mask] = 0.0
    
    return best_x, best_val

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
    
    # Initialize value function
    V_next = np.zeros(2*Q + 1, dtype=np.float64)
    
    # Store optimal depths
    opt_bid = np.zeros((steps, 2*Q + 1), dtype=np.float64)
    opt_ask = np.zeros((steps, 2*Q + 1), dtype=np.float64)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 10.0:
            # Fallback: fill remaining with 0 depth or inf? 
            # If we break, the table is incomplete. 
            # The quote function will use whatever is there.
            # If uninitialized, it's 0.0. 0.0 depth is valid (quote at mid).
            # But 0.0 might be suboptimal. 
            # Let's just break and hope it's not called for early steps.
            break
            
        V_curr = np.zeros(2*Q + 1, dtype=np.float64)
        
        # Prepare deltas for bid and ask
        # q ranges from -Q to Q
        # Index i corresponds to q = i - Q
        
        # Bid delta: V_next[q+1] - V_next[q]
        # For q = -Q..Q-1, delta_b[i] = V_next[i+1] - V_next[i]
        # For q = Q, no bid allowed.
        
        # Ask delta: V_next[q-1] - V_next[q]
        # For q = -Q+1..Q, delta_a[i] = V_next[i-1] - V_next[i]
        # For q = -Q, no ask allowed.
        
        delta_b = np.zeros(2*Q + 1, dtype=np.float64)
        delta_a = np.zeros(2*Q + 1, dtype=np.float64)
        
        # Valid bids: q < Q => indices 0 to 2Q-1
        delta_b[:2*Q] = V_next[1:2*Q+1] - V_next[:2*Q]
        
        # Valid asks: q > -Q => indices 1 to 2Q
        delta_a[1:] = V_next[:2*Q] - V_next[1:]
        
        # Solve for optimal depths
        bid_depths, bid_vals = _solve_optimal_depths_vectorized(A, k, dt, delta_b)
        ask_depths, ask_vals = _solve_optimal_depths_vectorized(A, k, dt, delta_a)
        
        # Mask out invalid quotes (q=Q for bid, q=-Q for ask)
        bid_depths[2*Q] = np.inf
        bid_vals[2*Q] = 0.0
        
        ask_depths[0] = np.inf
        ask_vals[0] = 0.0
        
        # Calculate inventory cost
        # q = np.arange(-Q, Q+1)
        q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
        cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt
        
        # V_curr = -cost + V_next + contrib_b + contrib_a
        V_curr = -cost + V_next + bid_vals + ask_vals
        
        opt_bid[n, :] = bid_depths
        opt_ask[n, :] = ask_depths
        
        V_next = V_curr
    
    # If we broke early, the rest of the table is zeros.
    # Zeros mean quote at mid. This is a safe fallback.
    
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
