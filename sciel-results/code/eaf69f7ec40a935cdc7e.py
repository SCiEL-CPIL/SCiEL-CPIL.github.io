import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0
_POLICY_CACHE = {}

def _solve_optimal_depth_vec(C_vals, alpha, k):
    """
    Vectorized solver for optimal depth x >= 0 maximizing (1 - exp(-alpha * exp(-k*x))) * (x + C)
    for each C in C_vals.
    Uses Newton's method with bisection safeguard.
    """
    C_vals = np.asarray(C_vals, dtype=float)
    n = len(C_vals)
    
    # Better initial guess: 
    # For large positive C, x should be small (high probability)
    # For large negative C, x should be large (low probability)
    # Rough heuristic: x0 = max(0, -C/k) * 0.5 + 1.0/k
    x = np.clip(-C_vals / k * 0.5, 0.0, 20.0 / k) + 1.0 / k
    
    # Bisection bounds
    x_lo = np.zeros(n)
    x_hi = np.full(n, 20.0 / k)
    
    for _ in range(80):
        exp_kx = np.exp(-k * x)
        v = alpha * exp_kx
        p = 1.0 - np.exp(-v)
        
        # f'(x) = p' * (x + C) + p
        # p' = -alpha * k * exp(-kx) * exp(-alpha * exp(-kx)) = -k * v * exp(-v)
        p_prime = -k * v * np.exp(-v)
        
        f_prime = p_prime * (x + C_vals) + p
        
        # f''(x) = p''(x)*(x+C) + 2*p'
        # p''(x) = k^2 * exp(-v) * (v - v^2)
        p_double_prime = k**2 * np.exp(-v) * (v - v**2)
        
        f_double_prime = p_double_prime * (x + C_vals) + 2 * p_prime
        
        # Newton step
        with np.errstate(divide='ignore', invalid='ignore'):
            dx = -f_prime / f_double_prime
        
        # Safeguard: if Newton step goes out of bounds or is too large, use bisection
        bad = (dx > (x_hi - x_lo) * 0.5) | (dx < -(x - x_lo) * 0.5) | (np.abs(dx) > 10.0 / k) | np.isnan(dx)
        
        # For bad points, use bisection
        x_mid = (x_lo + x_hi) / 2
        # Determine which side of the root based on sign of f_prime
        # If f_prime > 0, root is to the left (smaller x)
        # If f_prime < 0, root is to the right (larger x)
        go_left = f_prime > 0
        
        x_new = np.where(bad, np.where(go_left, x_lo, x_hi), x + dx)
        
        # Update bounds
        x_new = np.maximum(x_new, x_lo)
        x_new = np.minimum(x_new, x_hi)
        
        # Update bounds based on sign of f_prime at current x
        # If f_prime(x) > 0, the root is at smaller x, so x_hi = x
        # If f_prime(x) < 0, the root is at larger x, so x_lo = x
        x_hi = np.where(f_prime > 0, x, x_hi)
        x_lo = np.where(f_prime < 0, x, x_lo)
        
        # Check convergence
        if np.all(np.abs(x_new - x) < 1e-10):
            x = x_new
            break
        x = x_new
    
    return x

def _compute_policy(params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
        
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    alpha = A * dt
    
    n_states = 2 * Q + 1
    # V[n, q_idx]
    V = np.zeros((steps + 1, n_states))
    
    bid_depths = np.full((steps, n_states), np.inf)
    ask_depths = np.full((steps, n_states), np.inf)
    
    # Precompute q values and costs
    q_vals = np.arange(-Q, Q + 1)
    cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 2.0:
            # Emergency fallback
            for i in range(n, steps):
                bid_depths[i] = 1.0/k
                ask_depths[i] = 1.0/k
            _POLICY_CACHE[key] = (bid_depths, ask_depths)
            return _POLICY_CACHE[key]

        V_next = V[n+1]
        V_curr = V[n]
        
        # Calculate C_b and C_a for all valid q
        # C_b[q] = V_next[q+1] - V_next[q]
        # Valid for q < Q (indices 0 to 2Q-1)
        C_b = V_next[1:] - V_next[:-1]
        
        # C_a[q] = V_next[q-1] - V_next[q]
        # Valid for q > -Q (indices 1 to 2Q)
        C_a = V_next[:-1] - V_next[1:]
        
        # Solve for optimal depths directly
        b_opt = _solve_optimal_depth_vec(C_b, alpha, k)
        a_opt = _solve_optimal_depth_vec(C_a, alpha, k)
        
        # Store optimal depths
        bid_depths[n, :2*Q] = b_opt
        bid_depths[n, 2*Q] = np.inf
        
        ask_depths[n, 1:] = a_opt
        ask_depths[n, 0] = np.inf
        
        # Calculate probabilities and gains
        p_b = np.zeros(n_states)
        p_a = np.zeros(n_states)
        
        p_b[:2*Q] = 1.0 - np.exp(-alpha * np.exp(-k * b_opt))
        p_a[1:] = 1.0 - np.exp(-alpha * np.exp(-k * a_opt))
        
        gain_b = np.zeros(n_states)
        gain_a = np.zeros(n_states)
        
        gain_b[:2*Q] = p_b[:2*Q] * (b_opt + C_b)
        gain_a[1:] = p_a[1:] * (a_opt + C_a)
        
        # V_curr = -cost + V_next + gain_b + gain_a
        V_curr[:] = -cost + V_next + gain_b + gain_a

    _POLICY_CACHE[key] = (bid_depths, ask_depths)
    return _POLICY_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Returns (bid_depths, ask_depths) arrays of shape (4000,)
    """
    bid_table, ask_table = _compute_policy(params)
    
    steps = params["steps"]
    T = params["T"]
    dt = T / steps
    
    n = int(t / dt)
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    Q = params["Q"]
    
    inv_idx = np.clip(inventory + Q, 0, 2 * Q)
    
    bid_depths = bid_table[n, inv_idx]
    ask_depths = ask_table[n, inv_idx]
    
    return bid_depths.astype(float), ask_depths.astype(float)
