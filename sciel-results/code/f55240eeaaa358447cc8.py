"""Combined model-and-optimize for Erdos minimum overlap."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def objective(h):
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def objective_smooth(h, beta=100.0):
    c = np.correlate(h, 1.0 - h, mode="full")
    c_max = np.max(c)
    c_shifted = c - c_max
    s = np.sum(np.exp(beta * c_shifted))
    return (c_max + (1.0/beta) * np.log(s)) / N * 2.0

def project(h):
    """Project to [0,1]^N and fix sum to N/2."""
    h = np.asarray(h, dtype=np.float64).copy()
    target = N / 2.0
    
    # Bisection to find the right offset
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_trial = np.clip(h + mid, 0.0, 1.0)
        s = h_trial.sum()
        if s < target:
            lo = mid
        else:
            hi = mid
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    
    # Final fine-tuning
    for _ in range(20):
        s = h.sum()
        if abs(s - target) < 1e-14:
            break
        diff = target - s
        free = (h > 1e-15) & (h < 1.0 - 1e-15)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
        else:
            break
    return h

def fourier_h(theta, n_harm):
    """theta: [a0, (f1,p1,a1), (f2,p2,a2), ...] -> h vector."""
    h = np.full(N, 0.5)
    h = h + theta[0]
    idx = 1
    i = np.arange(N)
    for k in range(n_harm):
        if idx + 3 > len(theta):
            break
        f, p, a = theta[idx], theta[idx+1], theta[idx+2]
        h = h + a * np.sin(2*np.pi*f*i/N + p)
        idx += 3
    return h

def optimize_fourier(theta0, n_harm, max_iter=2000):
    def obj(x):
        h = project(fourier_h(x, n_harm))
        return objective_smooth(h, beta=50.0)
    try:
        res = minimize(obj, theta0, method="L-BFGS-B",
                       options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50})
        return res.x
    except Exception:
        return theta0

def main_search():
    best_h = np.full(N, 1.0)  # start with a pattern: high/low alternating? No, 0.5 seed
    # Actually, start with the seed 0.5
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)  # 1.0

    rng = np.random.default_rng(0)

    # Strategy 1: Fourier parameterization with multi-start
    # Use more harmonics and more trials
    n_harm = 12
    dim = 1 + 3 * n_harm
    
    # Try a few specific initial structures first
    # 1. Fundamental frequency
    i = np.arange(N)
    for freq in [1, 2, 3, 5, 10, 25]:
        if remaining() < 5.0:
            break
        for amp in [0.1, 0.2, 0.3, 0.4, 0.45]:
            if remaining() < 2.0:
                break
            # Initialize theta with this frequency
            theta0 = np.zeros(dim)
            theta0[1] = freq  # f1
            theta0[2] = 0.0   # p1
            theta0[3] = amp   # a1
            theta_opt = optimize_fourier(theta0, n_harm, max_iter=1000)
            h = project(fourier_h(theta_opt, n_harm))
            v = objective(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # 2. Random Fourier multi-start
    for trial in range(100):
        if remaining() < 10.0:
            break
        theta0 = np.zeros(dim)
        theta0[0] = rng.uniform(-0.05, 0.05)
        for k in range(n_harm):
            # Choose integer-ish frequencies
            # Bias towards lower frequencies for smoother patterns
            max_f = max(1, N//2 - k)
            theta0[1+3*k]   = rng.integers(1, max_f+1)
            theta0[2+3*k]   = rng.uniform(0, 2*np.pi)
            # Amplitudes decay with frequency
            theta0[3+3*k]   = rng.uniform(-0.5, 0.5) / (k + 1)
        
        theta_opt = optimize_fourier(theta0, n_harm, max_iter=1500)
        h = project(fourier_h(theta_opt, n_harm))
        v = objective(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
        
        if remaining() < 20.0:
            break

    # Strategy 2: Direct optimization on h with bounds
    # Use the best_h as starting point
    if remaining() > 30.0:
        def obj_direct(x):
            h = project(x)
            return objective_smooth(h, beta=50.0)
        
        try:
            res = minimize(obj_direct, best_h, method="L-BFGS-B",
                           bounds=[(0.0, 1.0)] * N,
                           options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
            h = project(res.x)
            v = objective(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
        except Exception:
            pass

    # Strategy 3: Local polish with random perturbations
    step_size = 0.05
    for i in range(3000):
        if remaining() < 5.0:
            break
        step_size = 0.05 * (1.0 - i/5000.0)
        if step_size < 0.001:
            step_size = 0.001
        h = best_h + rng.normal(0, step_size, N)
        h = project(h)
        v = objective(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Final high-precision polish
    if remaining() > 10.0:
        def obj_final(x):
            h = project(x)
            return objective_smooth(h, beta=100.0)
        
        try:
            res = minimize(obj_final, best_h, method="L-BFGS-B",
                           bounds=[(0.0, 1.0)] * N,
                           options={"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-12})
            h = project(res.x)
            v = objective(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
        except Exception:
            pass

    return best_h, best_val

# EVOLVE-BLOCK-START
def construct():
    global _CACHED
    try:
        if '_CACHED' in globals():
            return _CACHED
        h, _ = main_search()
        h = project(h)
        # enforce sum within tolerance
        s = h.sum()
        h = h + (N/2.0 - s)/N
        h = np.clip(h, 0.0, 1.0)
        assert np.all(np.isfinite(h))
        _CACHED = h
        return h
    except Exception:
        return np.full(N, 0.5)
# EVOLVE-BLOCK-END
