import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _enforce_constraints(h, target_sum=N/2.0):
    """Project h to satisfy sum constraint and [0,1] bounds."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        s = np.sum(h)
        diff = target_sum - s
        if abs(diff) < 1e-12:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _compute_all_overlaps(h):
    """Compute overlap for all shifts k: sum_i h_i * (1 - h_{i-k}) with zero padding."""
    overlaps = np.zeros(N)
    for k in range(N):
        if k == 0:
            overlaps[k] = np.sum(h * (1 - h))
        else:
            overlaps[k] = np.sum(h[k:] * (1 - h[:-k]))
    return overlaps

def _score(h):
    """Return the maximum overlap over all shifts."""
    overlaps = _compute_all_overlaps(h)
    return np.max(overlaps)

def _smooth_max(values, tau=0.01):
    """Smooth approximation of max using log-sum-exp."""
    v = np.asarray(values, dtype=float)
    m = np.max(v)
    return m + tau * np.log(np.sum(np.exp((v - m) / tau)))

def _smooth_max_grad(values, tau=0.01):
    """Gradient of smooth max w.r.t. values."""
    v = np.asarray(values, dtype=float)
    m = np.max(v)
    exp_v = np.exp((v - m) / tau)
    return exp_v / np.sum(exp_v)

def _sinusoidal_h(A, phi, N=N):
    """Generate sinusoidal step heights, projected to valid constraints."""
    i = np.arange(N)
    h = 0.5 + A * np.sin(2 * np.pi * i / N + phi)
    return _enforce_constraints(h)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Phase 1: Grid search over sinusoidal parameters
    best_h = None
    best_score = np.inf
    
    # Grid over amplitude and phase
    A_vals = np.linspace(0.0, 0.5, 20)
    phi_vals = np.linspace(0.0, 2 * np.pi, 40)
    
    for A in A_vals:
        for phi in phi_vals:
            if time.time() > deadline:
                break
            h = _sinusoidal_h(A, phi)
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
    
    if time.time() > deadline:
        return best_h
    
    # Phase 2: Local refinement using smooth max objective
    # Use scipy.optimize with analytic gradient
    
    def objective_and_grad(h):
        h = np.clip(h, 0.0, 1.0)
        overlaps = _compute_all_overlaps(h)
        tau = 0.02
        smooth_val = _smooth_max(overlaps, tau)
        
        # Gradient of smooth max w.r.t. overlaps
        grad_overlaps = _smooth_max_grad(overlaps, tau)
        
        # Gradient of overlaps w.r.t. h
        grad_h = np.zeros(N)
        for k in range(N):
            go = grad_overlaps[k]
            if go == 0:
                continue
            if k == 0:
                grad_h += go * (1 - 2 * h)
            else:
                # d/dh_j of sum_{i=k}^{N-1} h_i (1-h_{i-k})
                # Part 1: j in [k, N-1], h_j is first factor: (1-h_{j-k})
                # Part 2: j in [0, N-1-k], h_j is second factor: -h_{j+k}
                grad_h[k:] += go * (1 - h[:N-k])
                grad_h[:N-k] -= go * h[k:]
        
        # Project gradient onto sum=0 hyperplane
        grad_h -= np.mean(grad_h)
        
        return smooth_val, grad_h
    
    # Multiple restarts from different sinusoidal initializations
    restarts = [
        (0.3, 0.0), (0.3, np.pi/2), (0.3, np.pi), (0.3, 3*np.pi/2),
        (0.4, 0.0), (0.4, np.pi/2), (0.4, np.pi), (0.4, 3*np.pi/2),
        (0.2, 0.0), (0.2, np.pi/2), (0.2, np.pi), (0.2, 3*np.pi/2),
        (0.5, 0.0), (0.5, np.pi/2), (0.5, np.pi), (0.5, 3*np.pi/2),
    ]
    
    for A, phi in restarts:
        if time.time() > deadline:
            break
        
        h0 = _sinusoidal_h(A, phi)
        
        try:
            result = minimize(
                objective_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = _enforce_constraints(result.x)
            s = _score(h_opt)
            
            if s < best_score:
                best_score = s
                best_h = h_opt.copy()
        except Exception:
            continue
    
    if best_h is None:
        best_h = _sinusoidal_h(0.3, 0.0)
    
    return best_h
