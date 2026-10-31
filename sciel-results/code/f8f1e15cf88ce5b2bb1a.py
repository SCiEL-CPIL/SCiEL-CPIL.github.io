"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 15.0  # seconds to spare before 150s kill
TARGET_SUM = N / 2.0


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    if n == 0:
        return 0.0
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]  # shifted[x] = h[x+1] for x in [0,N-2]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target (default N/2), preserving [0,1] bounds.
    Uses iterative water-filling style correction for better bound handling."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float).copy()
    for _ in range(10):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        step = diff / len(h)
        h += step
        h = np.clip(h, 0.0, 1.0)
    # Final correction on non-saturated entries
    diff = target - h.sum()
    if abs(diff) > 1e-14:
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if mask.sum() > 0:
            h[mask] += diff / mask.sum()
            h = np.clip(h, 0.0, 1.0)
    return h


def _sinusoidal_h(A, phi, A2=0.0, phi2=0.0):
    """Generate step heights from sinusoidal model with optional 2nd harmonic.
    h_i = 0.5 + A*sin(2*pi*i/N + phi) + A2*sin(4*pi*i/N + phi2)"""
    i = np.arange(N)
    h = 0.5 + A * np.sin(2.0 * np.pi * i / N + phi)
    if abs(A2) > 1e-12:
        h += A2 * np.sin(4.0 * np.pi * i / N + phi2)
    return h


def _optimize_sinusoidal():
    """
    Optimize sinusoidal parameters (A, phi, A2, phi2) via:
    1. Coarse grid search over (A, phi)
    2. Local Nelder-Mead refinement on 4 params
    3. Multiple starting points
    """
    best_score = float('inf')
    best_h = None

    # Phase 1: Coarse grid search (1st harmonic only for speed)
    A_grid = np.linspace(0.05, 0.45, 25)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, 50)

    best_A = 0.25
    best_phi = 0.0
    best_grid_score = float('inf')

    for A in A_grid:
        for phi in phi_grid:
            h = _sinusoidal_h(A, phi)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            s = _score(h)
            if s < best_grid_score:
                best_grid_score = s
                best_A = A
                best_phi = phi

    # Phase 2: Local refinement with Nelder-Mead on 4 params
    def obj_sin(params):
        A, phi, A2, phi2 = params
        A = np.clip(A, 0.0, 0.5)
        A2 = np.clip(A2, -0.3, 0.3)
        h = _sinusoidal_h(A, phi, A2, phi2)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)

    x0 = np.array([best_A, best_phi, 0.0, 0.0])
    best_A2 = 0.0
    best_phi2 = 0.0
    try:
        res = minimize(obj_sin, x0, method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.fun < best_grid_score:
            best_grid_score = res.fun
            best_A = float(res.x[0])
            best_phi = float(res.x[1])
            best_A2 = float(res.x[2])
            best_phi2 = float(res.x[3])
    except Exception:
        pass

    # Phase 3: Try multiple starting points for local refinement
    for A_init in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for phi_init in [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]:
            for A2_init in [0.0, 0.05, -0.05]:
                x0 = np.array([A_init, phi_init, A2_init, 0.0])
                try:
                    res = minimize(obj_sin, x0, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
                    if res.fun < best_grid_score:
                        best_grid_score = res.fun
                        best_A = float(res.x[0])
                        best_phi = float(res.x[1])
                        best_A2 = float(res.x[2])
                        best_phi2 = float(res.x[3])
                except Exception:
                    pass

    # Build best sinusoidal h
    h_sin = _sinusoidal_h(best_A, best_phi, best_A2, best_phi2)
    h_sin = np.clip(h_sin, 0.0, 1.0)
    h_sin = _fix_sum(h_sin)
    best_score = _score(h_sin)
    best_h = h_sin.copy()

    return best_score, best_h, (best_A, best_phi, best_A2, best_phi2)


def _smooth_max_and_grad(x, beta):
    """
    Compute soft max of x and its gradient using log-sum-exp.
    Returns (value, gradient_vector).
    """
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-12:
        return 0.0, np.zeros_like(x)
    
    shifted = x - max_x
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    
    if wsum == 0:
        return max_x, np.zeros_like(x)
        
    value = max_x + math.log(wsum) / beta
    grad = weights / wsum
    return value, grad


def _objective_and_grad(h, beta, penalty_weight=100.0):
    """
    Compute the smooth objective value and its gradient.
    Objective: smooth_max(overlap) + penalty * (sum(h) - N/2)^2
    """
    n = len(h)
    h = np.clip(h, 0.0, 1.0)
    
    # Calculate overlap vector: o_i = h_i * (1 - h_{i+1})
    overlap = np.zeros(n)
    if n > 1:
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
        overlap[-1] = h[-1]
    else:
        overlap[0] = h[0]
        
    val_smooth, grad_overlap = _smooth_max_and_grad(overlap, beta)
    
    # Gradient of overlap w.r.t h
    # o_i depends on h_i and h_{i+1}
    # d(o_i)/d(h_i) = 1 - h_{i+1}  (for i < N-1)
    # d(o_i)/d(h_{i+1}) = -h_i      (for i < N-1)
    # d(o_{N-1})/d(h_{N-1}) = 1
    
    grad_h = np.zeros(n)
    if n > 1:
        grad_h[:-1] += grad_overlap[:-1] * (1.0 - h[1:])
        grad_h[1:] += grad_overlap[:-1] * (-h[:-1])
        grad_h[-1] += grad_overlap[-1] * 1.0
    else:
        grad_h[0] += grad_overlap[0] * 1.0
        
    # Penalty term for sum constraint
    sum_error = h.sum() - TARGET_SUM
    penalty_val = penalty_weight * sum_error * sum_error
    penalty_grad = 2.0 * penalty_weight * sum_error
    
    total_val = val_smooth + penalty_val
    total_grad = grad_h + penalty_grad
    
    return total_val, total_grad


def _optimize_direct(h0, deadline, beta_start=50.0, beta_end=500.0, max_iter=500):
    """
    Optimize h using L-BFGS-B with analytic gradients and adaptive beta.
    """
    h = h0.copy()
    h = _fix_sum(h)
    current_score = _score(h)
    
    stages = [
        (beta_start, max_iter // 2),
        (beta_end // 2, max_iter // 4),
        (beta_end, max_iter // 4)
    ]
    
    for beta, iters in stages:
        if time.time() > deadline - 2.0:
            break
            
        def obj_func(x):
            val, _ = _objective_and_grad(x, beta)
            return val
        
        def grad_func(x):
            _, grad = _objective_and_grad(x, beta)
            return grad
        
        try:
            res = minimize(
                obj_func, 
                h, 
                jac=grad_func,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            h_new = res.x
            h_new = _fix_sum(h_new)
            new_score = _score(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
        except Exception:
            pass
            
    return current_score, h


def _local_search_polish(h, max_iters=5000, deadline=None):
    """
    Direct local search: try adjusting individual entries or swapping adjacent pairs.
    Uses incremental overlap computation for speed.
    """
    h = h.copy()
    current_score = _score(h)
    improved = True
    iters = 0
    deltas = [0.01, -0.01, 0.02, -0.02, 0.005, -0.005, 0.002, -0.002]

    if deadline is None:
        deadline = time.time() + 30.0

    while improved and iters < max_iters:
        if time.time() > deadline:
            break
        improved = False
        iters += 1

        # Try small adjustments to each entry
        for i in range(N):
            for delta in deltas:
                new_val = h[i] + delta
                if new_val < 0 or new_val > 1:
                    continue
                h_trial = h.copy()
                h_trial[i] = new_val
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-12:
                    h = h_trial
                    current_score = s
                    improved = True
                    break

        # Try swapping adjacent entries
        for i in range(N - 1):
            if abs(h[i] - h[i+1]) < 1e-10:
                continue
            h_trial = h.copy()
            h_trial[i], h_trial[i+1] = h_trial[i+1], h_trial[i]
            h_trial = _fix_sum(h_trial)
            s = _score(h_trial)
            if s < current_score - 1e-12:
                h = h_trial
                current_score = s
                improved = True

    return current_score, h


def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)

    # Phase 1: Sinusoidal optimization
    best_score = float('inf')
    best_h = None

    if time.time() < deadline:
        s_sin, h_sin, _ = _optimize_sinusoidal()
        if s_sin < best_score:
            best_score = s_sin
            best_h = h_sin.copy()

    # Phase 2: Direct gradient-based optimization from best candidate
    if best_h is not None and time.time() < deadline - 20.0:
        s_direct, h_direct = _optimize_direct(best_h, deadline=deadline - 5.0, max_iter=1000)
        if s_direct < best_score:
            best_score = s_direct
            best_h = h_direct.copy()

    # Phase 3: Local search polish
    if best_h is not None and time.time() < deadline - 10.0:
        s_polish, h_polish = _local_search_polish(best_h, max_iters=3000, deadline=deadline - 3.0)
        if s_polish < best_score:
            best_score = s_polish
            best_h = h_polish.copy()

    # Phase 4: Final high-precision direct optimization
    if best_h is not None and time.time() < deadline - 3.0:
        s_final, h_final = _optimize_direct(best_h, deadline=deadline - 1.0, beta_start=100.0, beta_end=1000.0, max_iter=500)
        if s_final < best_score:
            best_score = s_final
            best_h = h_final.copy()

    # Ensure valid output
    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)

    return best_h
