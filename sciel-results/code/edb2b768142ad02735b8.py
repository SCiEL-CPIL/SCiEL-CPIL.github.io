"""Rewritten program for Erdos minimum overlap using spectral optimization."""

import time
import numpy as np
from scipy.optimize import minimize

N = 100

def _linear_overlap_max(h):
    """Calculate max linear overlap for shifts k=1..N-1."""
    max_ov = 0.0
    for k in range(1, N):
        ov = np.sum(h[k:] * (1.0 - h[:-k]))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _objective(params):
    """
    Objective function for optimization.
    params: array of length N+1 where params[0] is offset, params[1:] are h values.
    We use a soft-max approach to make the objective smooth.
    """
    h = params[1:]
    # Soft max of overlaps with temperature
    overlaps = np.array([np.sum(h[k:] * (1.0 - h[:-k])) for k in range(1, N)])
    # Use log-sum-exp for soft max
    max_ov = np.max(overlaps)
    if max_ov < 1e-10:
        return max_ov
    # Temperature for soft max
    temp = 0.1
    soft_max = max_ov + temp * np.log(np.sum(np.exp((overlaps - max_ov) / temp)))
    return soft_max

def _constr_sum(h):
    """Constraint: sum(h) = N/2"""
    return np.sum(h) - N / 2.0

def _project_to_feasible(h):
    """Project h to satisfy sum constraint and bounds [0,1]."""
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    if s > 0:
        h = h * (N / 2.0) / s
    h = np.clip(h, 0.0, 1.0)
    # Final adjustment to ensure exact sum
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-10:
        # Distribute difference to elements not at boundary
        mask = (h > 1e-6) & (h < 1.0 - 1e-6)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            # Re-normalize if needed
            s2 = np.sum(h)
            if s2 > 0 and abs(s2 - N/2.0) > 1e-10:
                h = h * (N / 2.0) / s2
                h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Baseline: constant 0.5
    best_h = np.full(N, 0.5)
    best_score = _linear_overlap_max(best_h)
    
    # Multi-mode spectral search
    # Try combinations of 1-3 cosine modes
    max_modes = 3
    max_amp = 0.45  # Keep amplitudes moderate to stay in [0,1]
    
    # First, find best single mode
    best_single = None
    best_single_score = best_score
    for m in range(1, N // 2 + 1):
        if time.time() > deadline - 30.0:
            break
        # Grid search for amplitude
        for A in np.linspace(0.05, max_amp, 30):
            i = np.arange(N)
            h = 0.5 + A * np.cos(2 * np.pi * m * i / N)
            h = np.clip(h, 0.0, 1.0)
            # Check sum
            s = np.sum(h)
            if abs(s - N/2.0) > 0.1:
                continue
            score = _linear_overlap_max(h)
            if score < best_single_score:
                best_single_score = score
                best_single = (m, A)
    
    if best_single is not None:
        m, A = best_single
        i = np.arange(N)
        best_h = 0.5 + A * np.cos(2 * np.pi * m * i / N)
        best_h = np.clip(best_h, 0.0, 1.0)
        best_score = best_single_score
    
    # Try two-mode combinations with the best single mode
    if best_single is not None and time.time() < deadline - 25.0:
        m1, A1 = best_single
        for m2 in range(1, N // 2 + 1):
            if m2 == m1:
                continue
            if time.time() > deadline - 20.0:
                break
            for A2 in np.linspace(0.02, max_amp * 0.5, 15):
                i = np.arange(N)
                h = 0.5 + A1 * np.cos(2 * np.pi * m1 * i / N) + A2 * np.cos(2 * np.pi * m2 * i / N)
                h = np.clip(h, 0.0, 1.0)
                s = np.sum(h)
                if abs(s - N/2.0) > 0.5:
                    continue
                score = _linear_overlap_max(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Local refinement using scipy optimization
    if time.time() < deadline - 10.0:
        # Use the current best as starting point
        h0 = best_h.copy()
        
        # Try multiple random perturbations and optimize
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb
            if r == 0:
                h_start = h0.copy()
            else:
                h_start = h0 + np.random.normal(0, 0.05, N)
                h_start = _project_to_feasible(h_start)
            
            # Define optimization with bounds
            bounds = [(0.0, 1.0)] * N
            constraints = [{'type': 'eq', 'fun': lambda x: _constr_sum(x)}]
            
            try:
                result = minimize(
                    _objective,
                    np.concatenate([[0.0], h_start]),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                if result.success:
                    h_opt = result.x[1:]
                    h_opt = _project_to_feasible(h_opt)
                    score = _linear_overlap_max(h_opt)
                    if score < best_score:
                        best_score = score
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final high-precision polish: coordinate descent on individual elements
    if time.time() < deadline - 2.0:
        for iteration in range(50):
            if time.time() > deadline - 1.0:
                break
            improved = False
            for i in range(N):
                old_val = best_h[i]
                # Try small perturbations
                for delta in [-0.01, -0.005, 0.005, 0.01]:
                    new_val = old_val + delta
                    if new_val < 0 or new_val > 1:
                        continue
                    test_h = best_h.copy()
                    test_h[i] = new_val
                    # Adjust sum
                    diff = N/2.0 - np.sum(test_h)
                    if abs(diff) > 1e-10:
                        # Distribute to other elements
                        mask = np.ones(N, dtype=bool)
                        mask[i] = False
                        mask &= (test_h > 1e-6) & (test_h < 1.0 - 1e-6)
                        if np.sum(mask) > 0:
                            test_h[mask] += diff / np.sum(mask)
                            test_h = np.clip(test_h, 0.0, 1.0)
                        else:
                            continue
                    score = _linear_overlap_max(test_h)
                    if score < best_score - 1e-10:
                        best_h = test_h
                        best_score = score
                        improved = True
                        break
            if not improved:
                break
    
    return best_h

if __name__ == "__main__":
    construct()
