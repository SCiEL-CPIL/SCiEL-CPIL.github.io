import numpy as np
import time

_cache = {}
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # 35s safety margin before 330s kill

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

    # Use a finer grid for better initial approximation
    n_grid = 8000
    x_grid = np.logspace(-15, 0, n_grid)
    log_x = np.log(x_grid)
    # p = 1 - exp(-lambda * x)
    p_grid = 1.0 - np.exp(-lambda_t * x_grid)

    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)

    # Pre-compute term for J: C - (1/k) * log_x
    inv_k = 1.0 / k
    log_x_scaled = inv_k * log_x
    
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
        q_vals = np.arange(-Q, Q + 1)
        cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt

        V_next = V[t + 1, :]

        # C_b[i] = V_next[i+1] - V_next[i] for i < 2Q (q < Q)
        # For q=Q (idx 2Q), C_b is not used / b=inf
        C_b = np.zeros(n_q)
        C_b[:2*Q] = V_next[1:] - V_next[:-1]

        # C_a[i] = V_next[i-1] - V_next[i] for i > 0 (q > -Q)
        # For q=-Q (idx 0), C_a is not used / a=inf
        C_a = np.zeros(n_q)
        C_a[1:] = V_next[:-1] - V_next[1:]

        # Compute J_b for all q, all grid points
        # Shape: (n_q, n_grid)
        J_b = p_grid[None, :] * (C_b[:, None] - log_x_scaled[None, :])

        # Find max
        max_idx_b = np.argmax(J_b, axis=1)
        max_val_b = np.max(J_b, axis=1)

        # Determine b values
        b_vals = -log_x_scaled[max_idx_b]

        # If max_val <= 0, it's better not to quote (b=inf)
        b_vals[max_val_b <= 1e-9] = np.inf

        # Boundary: q=Q (idx 2Q) must have b=inf
        b_vals[2*Q] = np.inf

        opt_b[t, :] = b_vals

        # Same for Ask
        J_a = p_grid[None, :] * (C_a[:, None] - log_x_scaled[None, :])

        max_idx_a = np.argmax(J_a, axis=1)
        max_val_a = np.max(J_a, axis=1)

        a_vals = -log_x_scaled[max_idx_a]
        a_vals[max_val_a <= 1e-9] = np.inf
        a_vals[0] = np.inf # q=-Q

        opt_a[t, :] = a_vals

        # Update V
        contrib_b = np.where(opt_b[t, :] == np.inf, 0.0, max_val_b)
        contrib_a = np.where(opt_a[t, :] == np.inf, 0.0, max_val_a)

        V[t, :] = V_next + cost + contrib_b + contrib_a

    # Final polish: refine depths using local search around grid-optimal x
    # Use more iterations and tighter tolerance for higher precision
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 30:  # only if we have time
        for t in range(N):
            if time.time() > _DEADLINE - 10:
                break
            V_next = V[t + 1, :] if t < N - 1 else np.zeros(n_q)

            C_b = np.zeros(n_q)
            C_b[:2*Q] = V_next[1:] - V_next[:-1]
            C_a = np.zeros(n_q)
            C_a[1:] = V_next[:-1] - V_next[1:]

            # Refine bid depths
            for i in range(2 * Q):
                b_cur = opt_b[t, i]
                if np.isinf(b_cur) or b_cur <= 0:
                    continue
                x_cur = np.exp(-k * b_cur)
                C = C_b[i]
                # 10 Newton iterations with tighter tolerance
                for _ in range(10):
                    p_cur = 1.0 - np.exp(-lambda_t * x_cur)
                    exp_term = np.exp(-lambda_t * x_cur)
                    dJdx = lambda_t * exp_term * (C - inv_k * np.log(x_cur)) + p_cur * (-1.0 / (k * x_cur))
                    if abs(dJdx) < 1e-14:
                        break
                    # Second derivative for Newton step
                    d2Jdx2 = lambda_t**2 * exp_term * inv_k / x_cur + lambda_t * exp_term * (-inv_k / x_cur**2) + p_cur * (1.0 / (k * x_cur**2))
                    if abs(d2Jdx2) < 1e-14:
                        break
                    x_new = x_cur - dJdx / d2Jdx2
                    # Clamp to valid range
                    if x_new <= 0:
                        x_new = 1e-15
                    if x_new > 1.0:
                        x_new = 1.0
                    if abs(x_new - x_cur) < 1e-12:
                        break
                    x_cur = x_new
                b_new = -np.log(x_cur) / k
                if 0 < b_new < 100:
                    opt_b[t, i] = b_new

            # Refine ask depths
            for i in range(1, n_q):
                a_cur = opt_a[t, i]
                if np.isinf(a_cur) or a_cur <= 0:
                    continue
                x_cur = np.exp(-k * a_cur)
                C = C_a[i]
                for _ in range(10):
                    p_cur = 1.0 - np.exp(-lambda_t * x_cur)
                    exp_term = np.exp(-lambda_t * x_cur)
                    dJdx = lambda_t * exp_term * (C - inv_k * np.log(x_cur)) + p_cur * (-1.0 / (k * x_cur))
                    if abs(dJdx) < 1e-14:
                        break
                    d2Jdx2 = lambda_t**2 * exp_term * inv_k / x_cur + lambda_t * exp_term * (-inv_k / x_cur**2) + p_cur * (1.0 / (k * x_cur**2))
                    if abs(d2Jdx2) < 1e-14:
                        break
                    x_new = x_cur - dJdx / d2Jdx2
                    if x_new <= 0:
                        x_new = 1e-15
                    if x_new > 1.0:
                        x_new = 1.0
                    if abs(x_new - x_cur) < 1e-12:
                        break
                    x_cur = x_new
                a_new = -np.log(x_cur) / k
                if 0 < a_new < 100:
                    opt_a[t, i] = a_new

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
