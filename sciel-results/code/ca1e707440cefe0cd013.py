import numpy as np
import time

_cache = {}
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 40s safety margin before 330s kill

def _solve_regime(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]

    # Deadline check: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        sigma = params['sigma']
        Q = params['Q']
        n_q = 2 * Q + 1
        N = params['steps']
        # Fallback: symmetric small spreads
        opt_b = np.full((N, n_q), 0.05)
        opt_a = np.full((N, n_q), 0.05)
        opt_b[:, 2 * Q] = np.inf   # no bid at max inventory
        opt_a[:, 0] = np.inf       # no ask at min inventory
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
    lambda_t = A * dt
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # Grid for initial guess of x = exp(-k*b)
    # x is in (0, 1]. b is in [0, inf).
    # A finer grid helps initialization but Newton's method will refine it.
    # 100 points is sufficient for a good start.
    G = 100
    x_grid = np.logspace(-10, 0, G)
    log_x_grid = np.log(x_grid)
    
    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)

    inv_k = 1.0 / k

    for t in range(N - 1, -1, -1):
        # Deadline check inside the loop
        if time.time() > _DEADLINE:
            # Fill remaining steps with simple fallback
            for tt in range(t, -1, -1):
                opt_b[tt, :] = 0.05
                opt_a[tt, :] = 0.05
                opt_b[tt, 2 * Q] = np.inf
                opt_a[tt, 0] = np.inf
            _cache[key] = (opt_b, opt_a)
            return _cache[key]

        # Compute cost and next values
        cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt
        V_next = V[t + 1, :]

        # C_b[i] = V_next[i+1] - V_next[i] for i < 2Q (q < Q)
        C_b = np.zeros(n_q)
        C_b[:2*Q] = V_next[1:] - V_next[:-1]
        
        # C_a[i] = V_next[i-1] - V_next[i] for i > 0 (q > -Q)
        C_a = np.zeros(n_q)
        C_a[1:] = V_next[:-1] - V_next[1:]

        # --- Optimize Bid Depth ---
        # Objective J(x) = (1 - exp(-lambda_t * x)) * (C_b - inv_k * log(x))
        # We want to maximize J(x) for x in (0, 1]
        
        # 1. Grid search for initial guess
        # Shape: (n_q, G)
        # C_b[:, None] - inv_k * log_x_grid[None, :]
        term_b = C_b[:, None] - inv_k * log_x_grid[None, :]
        p_grid = 1.0 - np.exp(-lambda_t * x_grid[None, :])
        J_b_grid = p_grid * term_b
        
        # For q=Q (idx 2Q), C_b is 0 (unused), but we force b=inf later.
        # For other q, find argmax.
        max_idx_b = np.argmax(J_b_grid, axis=1)
        x_init_b = x_grid[max_idx_b]
        
        # 2. Newton's Method refinement
        # Vectorized over q
        x_cur = x_init_b.copy()
        
        # Mask for valid quotes (q < Q)
        valid_b = np.ones(n_q, dtype=bool)
        valid_b[2*Q] = False
        
        # Newton iterations
        for _ in range(20):
            # Compute derivative and second derivative
            # f(x) = (1 - e^{-Lx})(C - (1/k)ln x)
            # f'(x) = L e^{-Lx}(C - (1/k)ln x) - (1/kx)(1 - e^{-Lx})
            # f''(x) = ...
            
            exp_term = np.exp(-lambda_t * x_cur)
            p = 1.0 - exp_term
            log_x = np.log(x_cur)
            
            # f(x)
            # f'(x)
            term1 = lambda_t * exp_term * (C_b - inv_k * log_x)
            term2 = -inv_k * (1.0 / x_cur) * p
            df = term1 + term2
            
            # f''(x)
            # d/dx [ L e^{-Lx} (C - (1/k) ln x) ] = -L^2 e^{-Lx} (C - (1/k) ln x) + L e^{-Lx} (-1/(kx))
            # d/dx [ - (1/k) (1/x) (1 - e^{-Lx}) ] = (1/k) (1/x^2) (1 - e^{-Lx}) - (1/k)(1/x)(L e^{-Lx})
            # Sum:
            # -L^2 e^{-Lx} (C - (1/k) ln x) - (L/kx) e^{-Lx} + (1/kx^2)(1 - e^{-Lx}) - (L/kx) e^{-Lx}
            # = -L^2 e^{-Lx} (C - (1/k) ln x) - (2L/kx) e^{-Lx} + (1/kx^2)(1 - e^{-Lx})
            
            d2f = -lambda_t**2 * exp_term * (C_b - inv_k * log_x) - (2.0 * lambda_t * inv_k / x_cur) * exp_term + (inv_k / (x_cur**2)) * p
            
            # Newton step: x_new = x - f/f'
            # Use clipping to stay in (0, 1]
            with np.errstate(divide='ignore', invalid='ignore'):
                step = df / d2f
                x_new = x_cur - step
            
            # Clamp
            x_new = np.clip(x_new, 1e-12, 1.0)
            
            # Check convergence
            if np.all(np.abs(x_new - x_cur) < 1e-10):
                x_cur = x_new
                break
            x_cur = x_new
        
        # Calculate optimal b
        b_vals = -np.log(x_cur) / inv_k
        # If the max value on the grid was <= 0, it's better not to quote (b=inf)
        # Actually, if C_b is very negative, J(x) might be negative for all x.
        # The grid max J_b_grid gives the best value. If that is <= 0, quote inf.
        max_val_b = np.max(J_b_grid, axis=1)
        no_quote_b = max_val_b <= 1e-9
        
        b_vals[no_quote_b] = np.inf
        b_vals[2*Q] = np.inf # Boundary condition
        
        opt_b[t, :] = b_vals

        # --- Optimize Ask Depth ---
        term_a = C_a[:, None] - inv_k * log_x_grid[None, :]
        J_a_grid = p_grid * term_a
        
        max_idx_a = np.argmax(J_a_grid, axis=1)
        x_init_a = x_grid[max_idx_a]
        
        x_cur = x_init_a.copy()
        
        valid_a = np.ones(n_q, dtype=bool)
        valid_a[0] = False
        
        for _ in range(20):
            exp_term = np.exp(-lambda_t * x_cur)
            p = 1.0 - exp_term
            log_x = np.log(x_cur)
            
            term1 = lambda_t * exp_term * (C_a - inv_k * log_x)
            term2 = -inv_k * (1.0 / x_cur) * p
            df = term1 + term2
            
            d2f = -lambda_t**2 * exp_term * (C_a - inv_k * log_x) - (2.0 * lambda_t * inv_k / x_cur) * exp_term + (inv_k / (x_cur**2)) * p
            
            with np.errstate(divide='ignore', invalid='ignore'):
                step = df / d2f
                x_new = x_cur - step
            
            x_new = np.clip(x_new, 1e-12, 1.0)
            
            if np.all(np.abs(x_new - x_cur) < 1e-10):
                x_cur = x_new
                break
            x_cur = x_new
        
        a_vals = -np.log(x_cur) / inv_k
        max_val_a = np.max(J_a_grid, axis=1)
        no_quote_a = max_val_a <= 1e-9
        
        a_vals[no_quote_a] = np.inf
        a_vals[0] = np.inf # Boundary condition
        
        opt_a[t, :] = a_vals

        # Update V
        contrib_b = np.where(opt_b[t, :] == np.inf, 0.0, np.max(J_b_grid, axis=1))
        # Note: The Newton step refines x, so the value J(x_new) might be slightly higher than J(x_grid).
        # We should use the refined value for V update.
        p_refined_b = 1.0 - np.exp(-lambda_t * x_cur)
        J_refined_b = p_refined_b * (C_b - inv_k * np.log(x_cur))
        # If no quote, contrib is 0.
        contrib_b = np.where(no_quote_b | (np.arange(n_q) == 2*Q), 0.0, J_refined_b)
        
        contrib_a = np.where(opt_a[t, :] == np.inf, 0.0, np.max(J_a_grid, axis=1))
        p_refined_a = 1.0 - np.exp(-lambda_t * x_cur)
        J_refined_a = p_refined_a * (C_a - inv_k * np.log(x_cur))
        contrib_a = np.where(no_quote_a | (np.arange(n_q) == 0), 0.0, J_refined_a)
        
        V[t, :] = V_next + cost + contrib_b + contrib_a

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
