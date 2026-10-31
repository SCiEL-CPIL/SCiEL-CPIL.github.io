"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=100.0):
    """Gradient of smooth max with respect to x."""
    x_max = np.max(x)
    exps = np.exp(alpha * (x - x_max))
    weights = exps / np.sum(exps)
    return weights

def _overlap_and_grad(h, alpha):
    """Compute overlap and gradient of smooth max w.r.t. h."""
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    smax = _smooth_max(overlap, alpha)
    w = _smooth_max_grad(overlap, alpha)
    
    grad_h = np.zeros(N)
    for i in range(N - 1):
        if w[i] > 1e-15:
            grad_h[i] += w[i] * (1.0 - h[i+1])
            grad_h[i+1] += w[i] * (-h[i])
    if w[N-1] > 1e-15:
        grad_h[N-1] += w[N-1]
    
    return overlap, smax, grad_h, w

def _project_to_feasible(h):
    """Project h to satisfy sum=N/2 and 0<=h<=1."""
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s < 1e-10:
        h = np.full(N, 0.5)
        return h
    h = h * (target / s)
    h = np.clip(h, 0.0, 1.0)
    # Fix residual iteratively
    for _ in range(20):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        else:
            mask = h > 0.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        h = np.clip(h, 0.0, 1.0)
    return h

def _two_block_init():
    """Analytic 2-block initial guess."""
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382
    B = 1.0 - A  # ~0.618
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    return h0

def _sinusoidal_h(A, phi, N=N):
    """Generate sinusoidal step heights, projected to [0,1] with sum=N/2."""
    x = np.arange(N)
    h = 0.5 + A * np.sin(2.0 * np.pi * x / N + phi)
    return _project_to_feasible(h)

def _sinusoidal_score(params):
    """Score for sinusoidal parameters [A, phi]."""
    A, phi = params
    h = _sinusoidal_h(A, phi)
    return _score(h)

def _optimize_sinusoidal(deadline):
    """Optimize sinusoidal parameters A and phi via grid search + local refinement."""
    best_score = float('inf')
    best_A = 0.0
    best_phi = 0.0
    
    # Grid search over A in [0, 0.5] and phi in [0, 2*pi)
    n_A = 50
    n_phi = 100
    for i in range(n_A):
        if time.time() > deadline - 30:
            break
        A = i * 0.5 / (n_A - 1)
        for j in range(n_phi):
            phi = 2.0 * np.pi * j / n_phi
            h = _sinusoidal_h(A, phi)
            sc = _score(h)
            if sc < best_score:
                best_score = sc
                best_A = A
                best_phi = phi
    
    # Local refinement around best grid point
    try:
        x0 = np.array([best_A, best_phi])
        result = minimize(
            _sinusoidal_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
        )
        if result.fun < best_score:
            best_score = result.fun
            best_A, best_phi = result.x
    except Exception:
        pass
    
    # Further refinement with tighter Nelder-Mead
    try:
        x0 = np.array([best_A, best_phi])
        result2 = minimize(
            _sinusoidal_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
        )
        if result2.fun < best_score:
            best_score = result2.fun
            best_A, best_phi = result2.x
    except Exception:
        pass
    
    h_best = _sinusoidal_h(best_A, best_phi)
    return h_best, best_score

def _slsqp_solve(h_init, t_init, alpha, maxiter, ftol, deadline):
    """Run SLSQP with epigraph formulation."""
    def objective_epigraph(params):
        h = params[:N]
        t = params[N]
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        smax = _smooth_max(overlap, alpha)
        return t + 1000.0 * np.maximum(0.0, smax - t)**2
    
    x0 = np.concatenate([h_init, [t_init]])
    bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
    
    def constraint_sum(params):
        return np.sum(params[:N]) - N/2.0
    
    def constraint_epigraph(params):
        h = params[:N]
        t = params[N]
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        return t - np.max(overlap)
    
    constraints_ext = [
        {'type': 'eq', 'fun': constraint_sum},
        {'type': 'ineq', 'fun': constraint_epigraph}
    ]
    
    result = minimize(
        objective_epigraph,
        x0,
        method='SLSQP',
        bounds=bounds_ext,
        constraints=constraints_ext,
        options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
    )
    
    if result.fun < 100.0:
        h_opt = result.x[:N].copy()
        h_opt = _project_to_feasible(h_opt)
        new_score = _score(h_opt)
        if np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
            return h_opt, new_score
    
    return None, float('inf')

def _gradient_descent(h_init, t_init, deadline):
    """Gradient-based local search with adaptive step size."""
    h = h_init.copy()
    best_score = _score(h)
    best_h = h.copy()
    t = best_score
    
    for iteration in range(10000):
        if time.time() > deadline - 10:
            break
        
        # Compute gradient of smooth max objective
        alpha = 500.0
        overlap, smax, grad_h, w = _overlap_and_grad(h, alpha)
        
        # Project gradient onto feasible subspace (sum constraint)
        # Use simple projection: subtract mean to maintain sum
        grad_centered = grad_h - np.mean(grad_h)
        
        # Line search with backtracking
        step_size = max(0.00005, 0.05 * (1.0 - iteration / 10000.0) ** 0.5)
        improved = False
        
        for shrink in range(20):
            trial_step = step_size * (0.5 ** shrink)
            if trial_step < 1e-10:
                break
            h_trial = h - trial_step * grad_centered
            h_trial = np.clip(h_trial, 0.0, 1.0)
            h_trial = _project_to_feasible(h_trial)
            sc = _score(h_trial)
            
            if sc < best_score - 1e-15:
                best_score = sc
                best_h = h_trial.copy()
                h = h_trial.copy()
                improved = True
                break
        
        if not improved and iteration % 100 == 0:
            # Try random perturbation
            rng = np.random.default_rng(iteration)
            h_trial = h.copy()
            idxs = rng.choice(N, size=5, replace=False)
            for i in idxs:
                h_trial[i] = np.clip(h_trial[i] + rng.uniform(-0.01, 0.01), 0.0, 1.0)
            h_trial = _project_to_feasible(h_trial)
            sc = _score(h_trial)
            if sc < best_score - 1e-15:
                best_score = sc
                best_h = h_trial.copy()
                h = h_trial.copy()
    
    return best_h, best_score

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with 2-block solution
    h0 = _two_block_init()
    current_score = _score(h0)
    h_final = h0.copy()
    
    # Phase 0: Sinusoidal parameter optimization (low-dimensional smooth landscape)
    try:
        if time.time() < deadline - 60.0:
            h_sin, sc_sin = _optimize_sinusoidal(deadline)
            if sc_sin < current_score:
                h_final = h_sin.copy()
                current_score = sc_sin
    except Exception:
        pass
    
    # Phase 1: SLSQP with epigraph formulation (moderate alpha)
    try:
        if time.time() < deadline - 30.0:
            h_opt, sc_opt = _slsqp_solve(h_final, current_score + 0.05, 200.0, 2000, 1e-14, deadline)
            if h_opt is not None and sc_opt < current_score:
                h_final = h_opt.copy()
                current_score = sc_opt
    except Exception:
        pass
    
    # Phase 2: High-precision polish with tighter alpha
    try:
        if time.time() < deadline - 15.0:
            h_opt, sc_opt = _slsqp_solve(h_final, current_score + 0.01, 500.0, 3000, 1e-16, deadline)
            if h_opt is not None and sc_opt < current_score:
                h_final = h_opt.copy()
                current_score = sc_opt
    except Exception:
        pass
    
    # Phase 3: Gradient-based local search refinement
    try:
        if time.time() < deadline - 10.0:
            h_grad, sc_grad = _gradient_descent(h_final, current_score, deadline)
            if sc_grad < current_score:
                h_final = h_grad.copy()
                current_score = sc_grad
    except Exception:
        pass
    
    # Phase 4: Try diverse structured initializations if time permits
    try:
        if time.time() < deadline - 20.0:
            rng2 = np.random.default_rng(123)
            
            # Sinusoidal patterns with various frequencies
            for freq in [1, 2, 3, 4, 5, 7, 10]:
                if time.time() > deadline - 15.0:
                    break
                for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
                    if time.time() > deadline - 15.0:
                        break
                    x = np.arange(N)
                    h_init = 0.5 + 0.4 * np.sin(2*np.pi*freq*x/N + phase)
                    h_init = _project_to_feasible(h_init)
                    
                    try:
                        h_s, sc_s = _slsqp_solve(h_init, _score(h_init) + 0.05, 300.0, 800, 1e-12, deadline)
                        if h_s is not None and sc_s < current_score:
                            h_final = h_s.copy()
                            current_score = sc_s
                    except Exception:
                        continue
    except Exception:
        pass
    
    # Final validation
    h_final = _project_to_feasible(h_final)
    final_score = _score(h_final)
    
    if not (np.all(h_final >= -1e-8) and np.all(h_final <= 1.0 + 1e-8)):
        h_final = np.full(N, 0.5)
    
    return h_final
