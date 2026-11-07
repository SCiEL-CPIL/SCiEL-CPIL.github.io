"""Erdos minimum overlap: minimize max correlation of h with 1-h."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def _correlation_max(h):
    """Compute max(np.correlate(h, 1-h, mode='full')) / N * 2"""
    c = np.correlate(h, 1.0 - h, mode='full')
    return np.max(c) / N * 2.0

def _objective(x):
    """Objective: maximize minimum margin to box constraints (i.e., minimize -min_margin).
    We want to minimize the max correlation, so we use a smooth approximation.
    Actually, let's directly minimize the max correlation using a soft-max approach."""
    h = x
    c = np.correlate(h, 1.0 - h, mode='full')
    # Use soft max with high temperature for sharpness
    t = 50.0
    # Shift to avoid overflow
    c_shifted = c - np.max(c)
    soft_max = np.max(c) + np.log(np.sum(np.exp(t * c_shifted))) / t
    return soft_max

def _objective_direct(x):
    """Direct max correlation objective."""
    h = x
    c = np.correlate(h, 1.0 - h, mode='full')
    return np.max(c)

def construct():
    start_time = time.time()
    
    # Seed: all 0.5
    h_seed = np.full(N, 0.5)
    best_h = h_seed.copy()
    best_val = _correlation_max(best_h)  # should be 0.5
    
    # Constraints: 0 <= h[i] <= 1, sum(h) = N/2
    bounds = [(0.0, 1.0)] * N
    A_eq = np.ones((1, N))
    b_eq = N / 2.0
    
    # Try several initial points and optimization methods
    initial_points = [
        h_seed.copy(),
        # Alternating 0 and 1
        np.where(np.arange(N) % 2 == 0, 0.0, 1.0),
        # Block pattern
        np.tile([0.0, 0.0, 1.0, 1.0], N // 4),
        # Cosine-based
        0.5 + 0.4 * np.cos(2 * np.pi * np.arange(N) / N),
        # Another cosine
        0.5 + 0.3 * np.sin(2 * np.pi * 2 * np.arange(N) / N),
    ]
    
    # Clip to valid range
    for i, ip in enumerate(initial_points):
        initial_points[i] = np.clip(ip, 0.0, 1.0)
        # Ensure sum constraint approximately
        s = np.sum(initial_points[i])
        if s > 0:
            initial_points[i] = initial_points[i] * (N / 2.0) / s
            initial_points[i] = np.clip(initial_points[i], 0.0, 1.0)
    
    # Try SLSQP with different starts
    for ip in initial_points:
        if time.time() - start_time > DEADLINE - 10:
            break
        try:
            res = minimize(
                _objective,
                ip,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                     'jac': lambda x: np.ones(N)}
                ],
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if res.success:
                h_try = np.clip(res.x, 0.0, 1.0)
                # Fix sum constraint
                diff = N/2.0 - np.sum(h_try)
                h_try = h_try + diff / N
                h_try = np.clip(h_try, 0.0, 1.0)
                val = _correlation_max(h_try)
                if val < best_val:
                    best_val = val
                    best_h = h_try.copy()
        except Exception:
            pass
    
    # Try L-BFGS-B with penalty for sum constraint
    if time.time() - start_time < DEADLINE - 15:
        for ip in initial_points[:3]:
            if time.time() - start_time > DEADLINE - 10:
                break
            try:
                def obj_with_penalty(x):
                    h = x
                    sum_penalty = (np.sum(h) - N/2.0) ** 2 * 100.0
                    return _objective(h) + sum_penalty
                
                res = minimize(
                    obj_with_penalty,
                    ip,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12}
                )
                h_try = np.clip(res.x, 0.0, 1.0)
                diff = N/2.0 - np.sum(h_try)
                h_try = h_try + diff / N
                h_try = np.clip(h_try, 0.0, 1.0)
                val = _correlation_max(h_try)
                if val < best_val:
                    best_val = val
                    best_h = h_try.copy()
            except Exception:
                pass
    
    # Final refinement with SLSQP from best found
    if time.time() - start_time < DEADLINE - 5:
        try:
            res = minimize(
                _objective,
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                     'jac': lambda x: np.ones(N)}
                ],
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res.success:
                h_try = np.clip(res.x, 0.0, 1.0)
                diff = N/2.0 - np.sum(h_try)
                h_try = h_try + diff / N
                h_try = np.clip(h_try, 0.0, 1.0)
                val = _correlation_max(h_try)
                if val < best_val:
                    best_val = val
                    best_h = h_try.copy()
        except Exception:
            pass
    
    # Final validation and fix
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N/2.0 - np.sum(best_h)
    best_h = best_h + diff / N
    best_h = np.clip(best_h, 0.0, 1.0)
    
    # Ensure finite
    if not np.all(np.isfinite(best_h)):
        return h_seed
    
    return best_h
