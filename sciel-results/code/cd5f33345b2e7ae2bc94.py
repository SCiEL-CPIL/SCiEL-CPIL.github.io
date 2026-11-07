import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _correlate_full(h, g):
    """Full cross-correlation of h and g, length 2N-1."""
    return np.correlate(h, g, mode='full')

def _compute_value(h):
    """The actual evaluator value."""
    corr = _correlate_full(h, 1.0 - h)
    return np.max(corr) / N * 2

def _objective(h):
    """Objective to minimize: max correlation value."""
    corr = _correlate_full(h, 1.0 - h)
    return np.max(corr)

def _objective_grad(h):
    """Return (max_corr, gradient) for the max-correlation objective. Vectorized."""
    h = np.asarray(h)
    g = 1.0 - h
    corr = _correlate_full(h, g)
    max_idx = np.argmax(corr)
    max_val = corr[max_idx]
    
    lag = max_idx - (N - 1)
    grad = np.zeros(N)
    
    # Vectorized: for each i, grad[i] = g[i-lag] (if valid) - h[i+lag] (if valid)
    if lag >= 0:
        # i - lag in [0, N-1] => i in [lag, N-1]
        grad[lag:] += g[:N - lag]
        # i + lag in [0, N-1] => i in [0, N-1-lag]
        grad[:N - lag] -= h[lag:]
    else:
        lag_abs = -lag
        # i - lag = i + lag_abs in [0, N-1] => i in [0, N-1-lag_abs]
        grad[:N - lag_abs] += g[lag_abs:]
        # i + lag = i - lag_abs in [0, N-1] => i in [lag_abs, N-1]
        grad[lag_abs:] -= h[:N - lag_abs]
    
    return max_val, grad

def _smooth_objective(h, temperature=1.0):
    """Log-sum-exp smooth approximation of max correlation."""
    corr = _correlate_full(h, 1.0 - h)
    m = np.max(corr)
    log_sum_exp = m + temperature * np.log(np.sum(np.exp((corr - m) / temperature)))
    return log_sum_exp

def _smooth_objective_grad(h, temperature=1.0):
    """Gradient of the smooth max correlation objective. Vectorized."""
    h = np.asarray(h)
    g = 1.0 - h
    corr = _correlate_full(h, g)
    
    m = np.max(corr)
    weights = np.exp((corr - m) / temperature)
    weights /= np.sum(weights)
    
    grad = np.zeros(N)
    for k in range(len(corr)):
        if weights[k] < 1e-15:
            continue
        lag = k - (N - 1)
        w = weights[k]
        if lag >= 0:
            grad[lag:] += w * g[:N - lag]
            grad[:N - lag] -= w * h[lag:]
        else:
            lag_abs = -lag
            grad[:N - lag_abs] += w * g[lag_abs:]
            grad[lag_abs:] -= w * h[:N - lag_abs]
    
    return np.sum(weights * corr), grad

def _epigraph_objective(x):
    """Epigraph: minimize t subject to corr_k <= t for all k, 0<=h<=1, sum(h)=N/2."""
    h = x[:N]
    t = x[N]
    corr = _correlate_full(h, 1.0 - h)
    return t

def _epigraph_grad(x):
    """Gradient of epigraph objective."""
    h = x[:N]
    t = x[N]
    corr = _correlate_full(h, 1.0 - h)
    max_idx = np.argmax(corr)
    
    grad = np.zeros(N + 1)
    grad[N] = 1.0  # dt/dt = 1
    
    lag = max_idx - (N - 1)
    g = 1.0 - h
    if lag >= 0:
        grad[lag:N] += g[:N - lag]
        grad[:N - lag] -= h[lag:]
    else:
        lag_abs = -lag
        grad[:N - lag_abs] += g[lag_abs:]
        grad[lag_abs:] -= h[:N - lag_abs]
    
    return t, grad

def _project_to_feasible(h):
    """Project h to satisfy 0<=h<=1 and sum(h)=N/2."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(500):
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-14:
            break
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.sum(mask) == 0:
            break
        step = diff / np.sum(mask)
        h[mask] += step
        h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-15:
        mask = (h > 1e-15) & (h < 1.0 - 1e-15)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h

def _adam_smooth(h0, temperature=0.5, max_iter=5000, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, time_limit=None):
    """Adam optimizer with smooth objective and projection."""
    h = h0.copy()
    m = np.zeros(N)
    v = np.zeros(N)
    t = 0
    start_time = time.time()
    best_h = h.copy()
    best_val = _objective(h)
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        t += 1
        val, grad = _smooth_objective_grad(h, temperature)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        h_new = h - lr * m_hat / (np.sqrt(v_hat) + eps)
        h_new = _project_to_feasible(h_new)
        
        new_val = _objective(h_new)
        if new_val < best_val:
            best_val = new_val
            best_h = h_new.copy()
        
        h = h_new
    
    return best_h, best_val

def _adam_exact(h0, max_iter=5000, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8, time_limit=None):
    """Adam optimizer with exact max objective and projection."""
    h = h0.copy()
    m = np.zeros(N)
    v = np.zeros(N)
    t = 0
    start_time = time.time()
    best_h = h.copy()
    best_val = _objective(h)
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        t += 1
        val, grad = _objective_grad(h)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        h_new = h - lr * m_hat / (np.sqrt(v_hat) + eps)
        h_new = _project_to_feasible(h_new)
        
        new_val = _objective(h_new)
        if new_val < best_val:
            best_val = new_val
            best_h = h_new.copy()
        
        h = h_new
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0, 'jac': lambda x: np.ones(N)}
    ]
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    starts = [np.full(N, 0.5)]
    
    for _ in range(8):
        h_pert = np.full(N, 0.5) + rng.randn(N) * 0.15
        starts.append(_project_to_feasible(h_pert))
    
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for phase in [0, 0.5, 1.0]:
            h_sine = 0.5 + 0.45 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase * np.pi)
            starts.append(_project_to_feasible(h_sine))
    
    h_ramp = np.linspace(0.0, 1.0, N)
    starts.append(_project_to_feasible(h_ramp))
    
    h_bin = np.zeros(N)
    h_bin[:50] = 1.0
    starts.append(h_bin.copy())
    
    for base in [0.2, 0.25, 0.3, 0.35]:
        h_alt = np.where(np.arange(N) % 2 == 0, base, 1.0 - base)
        starts.append(_project_to_feasible(h_alt))
    
    for pattern in [[0.0, 0.5, 1.0], [0.0, 0.4, 0.8], [0.1, 0.5, 0.9]]:
        h_three = np.tile(pattern, N // len(pattern) + 1)[:N]
        starts.append(_project_to_feasible(h_three))
    
    h_spread = np.full(N, 0.5)
    for i in range(0, N, 2):
        h_spread[i] = 0.75
        if i + 1 < N:
            h_spread[i+1] = 0.25
    starts.append(_project_to_feasible(h_spread))
    
    for freq in [1, 2, 3]:
        h_cos = 0.5 + 0.45 * np.cos(2 * np.pi * freq * np.arange(N) / N)
        starts.append(_project_to_feasible(h_cos))
    
    # Add more diverse starts: random permutations of block patterns
    for _ in range(5):
        h_rand = np.concatenate([np.full(25, 0.0), np.full(25, 1.0), np.full(25, 0.2), np.full(25, 0.8)])
        rng.shuffle(h_rand)
        starts.append(_project_to_feasible(h_rand))
    
    # Gradient-based starts from sine patterns
    for freq in [2, 3, 5]:
        h_s = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N)
        h_s = _project_to_feasible(h_s)
        # Perturb and project
        for _ in range(3):
            h_p = _project_to_feasible(h_s + rng.randn(N) * 0.05)
            starts.append(h_p)
    
    def try_slsqp(h0, max_iter=800, ftol=1e-10):
        nonlocal best_h, best_val
        try:
            res = minimize(
                _objective,
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': ftol, 'disp': False}
            )
            h_res = _project_to_feasible(res.x)
            val = _objective(h_res)
            if val < best_val:
                best_val = val
                best_h = h_res.copy()
                return True
        except Exception:
            pass
        return False
    
    def try_smooth_slsqp(h0, temperature=0.5, max_iter=500, ftol=1e-10):
        nonlocal best_h, best_val
        try:
            res = minimize(
                lambda x: _smooth_objective(x, temperature),
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': ftol, 'disp': False}
            )
            h_res = _project_to_feasible(res.x)
            val = _objective(h_res)
            if val < best_val:
                best_val = val
                best_h = h_res.copy()
                return True
        except Exception:
            pass
        return False
    
    # Phase 1: SLSQP on all starts
    for h0 in starts:
        if time.time() > deadline - 50:
            break
        try_slsqp(h0, max_iter=400, ftol=1e-10)
    
    # Phase 2: Adam smooth gradient descent on best solution
    if time.time() < deadline - 40:
        time_left = deadline - 40 - time.time()
        h_gd, val_gd = _adam_smooth(best_h.copy(), temperature=0.5, max_iter=5000, lr=0.01, time_limit=time_left)
        if val_gd < best_val:
            best_val = val_gd
            best_h = h_gd.copy()
    
    # Phase 3: Adam exact gradient descent refinement
    if time.time() < deadline - 25:
        time_left = deadline - 25 - time.time()
        h_gd2, val_gd2 = _adam_exact(best_h.copy(), max_iter=5000, lr=0.005, time_limit=time_left)
        if val_gd2 < best_val:
            best_val = val_gd2
            best_h = h_gd2.copy()
    
    # Phase 4: SLSQP polish on current best
    if time.time() < deadline - 15:
        try_slsqp(best_h.copy(), max_iter=1500, ftol=1e-12)
    
    # Phase 5: Smooth SLSQP with lower temperature
    if time.time() < deadline - 10:
        try_smooth_slsqp(best_h.copy(), temperature=0.1, max_iter=1000, ftol=1e-12)
    
    # Phase 6: Final Adam polish with very small learning rate
    if time.time() < deadline - 5:
        time_left = deadline - 5 - time.time()
        h_gd3, val_gd3 = _adam_exact(best_h.copy(), max_iter=3000, lr=0.001, time_limit=time_left)
        if val_gd3 < best_val:
            best_val = val_gd3
            best_h = h_gd3.copy()
    
    # Final projection
    best_h = _project_to_feasible(best_h)
    
    return best_h
