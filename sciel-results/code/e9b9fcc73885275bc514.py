import numpy as np
from scipy.optimize import minimize
import time

N = 100

def objective(h):
    corr = np.correlate(h, 1.0 - h, mode='full')
    return np.max(corr)

def objective_grad(h):
    """Return (max_corr, gradient) for the max-correlation objective."""
    h = np.asarray(h)
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode='full')
    max_idx = np.argmax(corr)
    max_val = corr[max_idx]
    
    # Gradient of correlate(h, 1-h) w.r.t. h at the maximizing lag
    # correlate(h, 1-h, 'full')[k] = sum_j h[j] * (1-h)[j+k-N+1] for appropriate k
    # We need d/dh_i of corr[max_idx]
    # For lag k in 'full' mode (0..2N-2), corr[k] = sum_j h[j] * (1-h)[j - (k - (N-1))]
    # Let lag = k - (N-1), so corr[k] = sum_j h[j] * (1-h)[j - lag]
    # d(corr[k])/dh_i = (1-h)[i-lag] + h[i] * (-1) * delta[i-lag == j] ... 
    # Actually: corr[k] = sum_j h[j] * (1-h)[j - lag]
    # d/dh_i: term where j=i gives (1-h)[i-lag], and term where j-lag=i i.e. j=i+lag gives h[i+lag]*(-1)
    # So grad_i = (1-h)[i-lag] - h[i+lag] (with bounds checking)
    
    lag = max_idx - (N - 1)
    grad = np.zeros(N)
    
    for i in range(N):
        # Contribution from h[i] * (1-h)[i-lag]
        j1 = i - lag
        if 0 <= j1 < N:
            grad[i] += (1.0 - h[j1])
        # Contribution from h[i+lag] * (1-h)[i] -> d/dh[i] of -h[i+lag]*h[i] = -h[i+lag]
        j2 = i + lag
        if 0 <= j2 < N:
            grad[i] += -h[j2]
    
    return max_val, grad

def smooth_objective(h, temperature=1.0):
    """Log-sum-exp smooth approximation of max correlation."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    # Scale to avoid overflow
    m = np.max(corr)
    log_sum_exp = m + np.log(np.sum(np.exp((corr - m) / temperature))) * temperature
    return log_sum_exp

def smooth_objective_grad(h, temperature=1.0):
    """Gradient of the smooth max correlation objective."""
    h = np.asarray(h)
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode='full')
    
    # Soft-max weights
    m = np.max(corr)
    weights = np.exp((corr - m) / temperature)
    weights /= np.sum(weights)
    
    # Gradient is weighted sum of gradients of each correlation lag
    grad = np.zeros(N)
    for k in range(len(corr)):
        if weights[k] < 1e-15:
            continue
        lag = k - (N - 1)
        for i in range(N):
            j1 = i - lag
            if 0 <= j1 < N:
                grad[i] += weights[k] * (1.0 - h[j1])
            j2 = i + lag
            if 0 <= j2 < N:
                grad[i] += weights[k] * (-h[j2])
    
    return np.sum(weights * corr), grad

def project_to_feasible(h):
    """Project h to satisfy 0<=h<=1 and sum(h)=N/2."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-12:
            break
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if np.sum(mask) == 0:
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin under 150s kill

    best_h = np.full(N, 0.5)
    best_val = objective(best_h)

    bounds = [(0.0, 1.0) for _ in range(N)]
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0, 'jac': lambda x: np.ones(N)}
    ]

    # Multiple starting points
    rng = np.random.RandomState(42)
    starts = [np.full(N, 0.5)]

    # Random perturbations
    for _ in range(5):
        h_pert = np.full(N, 0.5) + rng.randn(N) * 0.1
        starts.append(project_to_feasible(h_pert))

    # Structured starts
    # Sine waves with different frequencies
    for freq in [1, 2, 3, 4, 5]:
        h_sine = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N)
        starts.append(project_to_feasible(h_sine))

    # Linear ramp
    h_ramp = np.linspace(0.0, 1.0, N)
    starts.append(project_to_feasible(h_ramp))

    # Two-level: first 50 are 1, rest 0
    h_bin = np.zeros(N)
    h_bin[:50] = 1.0
    starts.append(h_bin.copy())

    # Alternating 0.25 and 0.75
    h_alt = np.where(np.arange(N) % 2 == 0, 0.25, 0.75)
    starts.append(h_alt.copy())

    # Three-level pattern
    h_three = np.tile([0.0, 0.5, 1.0], N // 3 + 1)[:N]
    starts.append(project_to_feasible(h_three))

    # Spread pattern: distribute mass evenly
    h_spread = np.full(N, 0.5)
    for i in range(0, N, 2):
        h_spread[i] = 0.7
        if i + 1 < N:
            h_spread[i+1] = 0.3
    starts.append(project_to_feasible(h_spread))

    def try_optimize(h0, max_iter=800, ftol=1e-10):
        """Run SLSQP optimization and return result if improved."""
        nonlocal best_h, best_val
        try:
            res = minimize(
                objective,
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': ftol, 'disp': False}
            )
            h_res = project_to_feasible(res.x)
            val = objective(h_res)
            if val < best_val:
                best_val = val
                best_h = h_res.copy()
                return True
        except Exception:
            pass
        return False

    def try_smooth_optimize(h0, temperature=0.5, max_iter=500, ftol=1e-10):
        """Run SLSQP with smooth objective."""
        nonlocal best_h, best_val
        try:
            res = minimize(
                lambda x: smooth_objective(x, temperature)[0],
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': ftol, 'disp': False}
            )
            h_res = project_to_feasible(res.x)
            val = objective(h_res)
            if val < best_val:
                best_val = val
                best_h = h_res.copy()
                return True
        except Exception:
            pass
        return False

    # Phase 1: Standard SLSQP on all starts
    for h0 in starts:
        if time.time() > deadline - 30:
            break
        try_optimize(h0, max_iter=800, ftol=1e-10)

    # Phase 2: Smooth-max refinement on best solution
    if time.time() < deadline - 20:
        try_smooth_optimize(best_h.copy(), temperature=0.5, max_iter=500, ftol=1e-10)

    # Phase 3: High-precision polish with tighter tolerance
    if time.time() < deadline - 10:
        try_optimize(best_h.copy(), max_iter=2000, ftol=1e-14)

    # Phase 4: Another smooth pass with lower temperature
    if time.time() < deadline - 5:
        try_smooth_optimize(best_h.copy(), temperature=0.1, max_iter=300, ftol=1e-12)

    # Final projection
    best_h = project_to_feasible(best_h)

    return best_h
