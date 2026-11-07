"""Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

N = 100
DEADLINE = 135.0
_START = time.time()


def remaining_time():
    return DEADLINE - (time.time() - _START)


def objective(h):
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def project(h):
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        s = h.sum()
        if abs(s - target) < 1e-14:
            break
        # Adjust elements to correct sum while respecting bounds
        excess = s - target
        if excess > 0:
            # Need to decrease sum; subtract from elements > 0
            mask = h > 0
            if not np.any(mask):
                break
            # Subtract proportionally, but respect lower bound 0
            avail = h[mask].sum()
            step = min(excess, avail)
            h[mask] -= step * h[mask] / avail
        else:
            # Need to increase sum; add to elements < 1
            mask = h < 1
            if not np.any(mask):
                break
            avail = (1 - h[mask]).sum()
            step = min(-excess, avail)
            h[mask] += step * (1 - h[mask]) / avail
        h = np.clip(h, 0.0, 1.0)
    # Final precise adjustment
    for _ in range(50):
        s = h.sum()
        diff = target - s
        if abs(diff) < 1e-15:
            break
        mask = (h > 0) & (h < 1)
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Generate diverse initializations
    inits = []
    # Cosine waves of various frequencies
    for k in range(1, 10):
        for phase in [0, np.pi/2, np.pi/3]:
            t = np.arange(N)
            h = 0.5 + 0.5 * np.cos(2 * np.pi * k * t / N + phase)
            inits.append(project(h))
    # Square waves
    for k in range(1, 6):
        t = np.arange(N)
        h = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * k * t / N))
        inits.append(project(h))
    # Random
    for _ in range(5):
        rng = np.random.default_rng(_ + 42)
        h = rng.uniform(0, 1, N)
        inits.append(project(h))
    
    # Try each initialization with L-BFGS-B
    for h0 in inits:
        if remaining_time() < 10:
            break
        
        def fun(x):
            hx = project(x)
            return objective(hx)
        
        result = minimize(fun, h0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-12})
        
        if result.success or result.fun < best_val:
            h_opt = project(result.x)
            val = objective(h_opt)
            if val < best_val:
                best_h = h_opt
                best_val = val
    
    # Local search refinement: try small perturbations
    if remaining_time() > 5:
        for _ in range(100):
            if remaining_time() < 5:
                break
            rng = np.random.default_rng(int(time.time() * 1000) % 10000)
            h_pert = best_h + rng.normal(0, 0.05, N)
            h_pert = project(h_pert)
            val = objective(h_pert)
            if val < best_val:
                best_h = h_pert
                best_val = val
    
    # Final projection to ensure constraints
    best_h = project(best_h)
    # Ensure sum is exactly correct
    s = best_h.sum()
    diff = N / 2.0 - s
    mask = (best_h > 0) & (best_h < 1)
    if np.any(mask):
        best_h[mask] += diff / np.sum(mask)
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
