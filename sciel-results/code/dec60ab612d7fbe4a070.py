import numpy as np
import time
from scipy.optimize import brentq
from scipy.interpolate import interp1d

_cache = {}
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _precompute_lut(A, k, dt, C_min=-10.0, C_max=10.0, n_lut=1000):
    """
    Pre-compute optimal depth d and max value J as a function of C.
    C is the marginal value (V(q+1)-V(q) or V(q-1)-V(q)).
    """
    L = A * dt
    inv_k = 1.0 / k
    
    C_grid = np.linspace(C_min, C_max, n_lut)
    d_lut = np.zeros(n_lut)
    J_lut = np.zeros(n_lut)
    
    # Function to maximize: f(x) = (1 - exp(-Lx)) * (C - log(x)/k)
    # x in (0, 1]
    # We solve for x, then convert to d = -log(x)/k
    
    # We can vectorize the root finding? No, brentq is scalar.
    # But n_lut=1000 is small.
    
    for i, C in enumerate(C_grid):
        def h(x):
            exp_term = np.exp(-L * x)
            term1 = L * exp_term * (C - inv_k * np.log(x))
            term2 = (1.0 - exp_term) * inv_k / x
            return term1 - term2
        
        # Find root of h(x) = 0 in (0, 1]
        # h(1) = L*C
        # h(0+) -> -inf (since -1/(kx) dominates)
        # If h(1) > 0, root is in (0, 1).
        # If h(1) <= 0, max is at x=1? 
        # Wait, if h(1) < 0, function is decreasing at 1.
        # f(1) = 0. f(0) = 0.
        # If it decreases at 1, it must have increased before.
        # So there is a root where h goes from + to -.
        # Actually, if h(1) < 0, the maximum is inside (0,1).
        # Let's check h(x) for small x.
        # If C is very negative, the max might be at very small x?
        # Let's just try to find a bracket.
        
        x_lo = 1e-12
        x_hi = 1.0
        
        try:
            f_lo = h(x_lo)
            f_hi = h(x_hi)
            
            if f_lo * f_hi > 0:
                # No sign change?
                # If f_hi > 0, max is at 1 (d=0)
                # If f_lo < 0 and f_hi < 0, max is inside?
                # Let's sample a few points to find a sign change.
                x_mid = np.logspace(-12, 0, 10)
                h_vals = h(x_mid)
                sign_changes = np.where(np.diff(np.sign(h_vals)) != 0)[0]
                if len(sign_changes) > 0:
                    idx = sign_changes[0]
                    x_lo = x_mid[idx]
                    x_hi = x_mid[idx+1]
                    x_star = brentq(h, x_lo, x_hi, xtol=1e-12, rtol=1e-12)
                else:
                    # Fallback: if h(1) > 0, x=1. Else x=1e-12?
                    if f_hi > 0:
                        x_star = 1.0
                    else:
                        # Find max of f directly on a small grid
                        x_grid = np.logspace(-12, 0, 100)
                        f_vals = (1 - np.exp(-L * x_grid)) * (C - inv_k * np.log(x_grid))
                        x_star = x_grid[np.argmax(f_vals)]
            else:
                x_star = brentq(h, x_lo, x_hi, xtol=1e-12, rtol=1e-12)
                
            d = -np.log(x_star) * inv_k
            J = (1.0 - np.exp(-L * x_star)) * (C - inv_k * np.log(x_star))
            
            d_lut[i] = d
            J_lut[i] = J
            
        except Exception:
            # Fallback to a reasonable guess
            d_lut[i] = 0.1
            J_lut[i] = 0.0

    # Create interpolators
    # Use linear interpolation for d and J
    interp_d = interp1d(C_grid, d_lut, kind='linear', bounds_error=False, fill_value=(d_lut[0], d_lut[-1]))
    interp_J = interp1d(C_grid, J_lut, kind='linear', bounds_error=False, fill_value=(J_lut[0], J_lut[-1]))
    
    return interp_d, interp_J

def _solve_regime(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]

    if time.time() > _DEADLINE:
        # Fallback
        sigma = params['sigma']
        Q = params['Q']
        n_q = 2 * Q + 1
        N = params['steps']
        opt_b = np.full((N, n_q), 0.05)
        opt_a = np.full((N, n_q), 0.05)
        opt_b[:, 2 * Q] = np.inf
        opt_a[:, 0] = np.inf
        _cache[key] = (opt_b, opt_a)
        return _cache[key]

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)

    cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt

    # Pre-compute LUT
    interp_d, interp_J = _precompute_lut(A, k, dt)
    
    # Iterate over time steps backwards
    for t in range(N - 1, -1, -1):
        if time.time() > _DEADLINE:
             # Fill remaining with fallback
             for tt in range(t, -1, -1):
                 opt_b[tt, :] = 0.05
                 opt_a[tt, :] = 0.05
                 opt_b[tt, 2 * Q] = np.inf
                 opt_a[tt, 0] = np.inf
             _cache[key] = (opt_b, opt_a)
             return _cache[key]

        V_next = V[t + 1, :]
        
        C_b = np.zeros(n_q)
        C_a = np.zeros(n_q)
        
        C_b[:2*Q] = V_next[1:] - V_next[:-1]
        C_a[1:] = V_next[:-1] - V_next[1:]

        # Get optimal depths from LUT
        d_b = interp_d(C_b)
        d_a = interp_d(C_a)
        
        J_b = interp_J(C_b)
        J_a = interp_J(C_a)
        
        # If J <= 0, don't quote
        b_vals = np.where(J_b > 1e-9, d_b, np.inf)
        a_vals = np.where(J_a > 1e-9, d_a, np.inf)
        
        # Handle boundaries
        b_vals[2*Q] = np.inf
        a_vals[0] = np.inf
        
        # Calculate probabilities
        x_b = np.exp(-k * b_vals)
        x_a = np.exp(-k * a_vals)
        
        # Handle inf depth -> x=0
        x_b = np.where(np.isinf(b_vals), 0.0, x_b)
        x_a = np.where(np.isinf(a_vals), 0.0, x_a)
        
        p_b = 1.0 - np.exp(-A * x_b * dt)
        p_a = 1.0 - np.exp(-A * x_a * dt)
        
        # Ensure p is 0 if depth is inf
        p_b = np.where(np.isinf(b_vals), 0.0, p_b)
        p_a = np.where(np.isinf(a_vals), 0.0, p_a)

        opt_b[t, :] = b_vals
        opt_a[t, :] = a_vals
        
        # Update V
        reward_b = p_b * (b_vals + C_b)
        reward_b = np.where(np.isinf(b_vals), 0.0, reward_b)
        
        reward_a = p_a * (a_vals + C_a)
        reward_a = np.where(np.isinf(a_vals), 0.0, reward_a)
        
        cross_term = p_b * p_a * (C_b + C_a)
        cross_term = np.where(np.isfinite(C_b + C_a), cross_term, 0.0)
        
        V[t, :] = V_next + cost + reward_b + reward_a - cross_term

    _cache[key] = (opt_b, opt_a)
    return _cache[key]

def quote(t, inventory, mid, params):
    opt_b, opt_a = _solve_regime(params)
    dt = params['T'] / params['steps']
    Q = params['Q']
    
    t_idx = int(t / dt)
    if t_idx >= len(opt_b):
        t_idx = len(opt_b) - 1
    if t_idx < 0:
        t_idx = 0
        
    idxs = inventory + Q
    
    bid_depths = opt_b[t_idx, idxs]
    ask_depths = opt_a[t_idx, idxs]
    
    return bid_depths, ask_depths
