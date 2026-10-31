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
    """Compute overlap, smooth max, and gradient of smooth max w.r.t. h."""
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
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
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
    """Optimize sinusoidal parameters via grid search + Nelder-Mead refinement."""
    best_score = float('inf')
    best_A = 0.0
    best_phi = 0.0
    
    # Grid search
    n_A = 40
    n_phi = 80
    for i in range(n_A):
        if time.time() > deadline - 40:
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
    
    # Nelder-Mead refinement
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
    
    # Tighter refinement
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

def _gradient_descent(h_init, deadline):
    """Gradient-based local search with adaptive step size and line search."""
    h = h_init.copy()
    best_score = _score(h)
    best_h = h.copy()
    
    for iteration in range(20000):
        if time.time() > deadline - 15:
            break
        
        alpha = 500.0
        overlap, smax, grad_h, w = _overlap_and_grad(h, alpha)
        
        # Project gradient onto feasible subspace
        grad_centered = grad_h - np.mean(grad_h)
        
        step_size = max(0.0001, 0.05 * (1.0 - iteration / 20000.0) ** 0.5)
        improved = False
        
        for shrink in range(25):
            trial_step = step_size * (0.5 ** shrink)
            if trial_step < 1e-12:
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
        
        if not improved and iteration % 200 == 0:
            rng = np.random.default_rng(iteration)
            h_trial = h.copy()
            idxs = rng.choice(N, size=3, replace=False)
            for i in idxs:
                h_trial[i] = np.clip(h_trial[i] + rng.uniform(-0.005, 0.005), 0.0, 1.0)
            h_trial = _project_to_feasible(h_trial)
            sc = _score(h_trial)
            if sc < best_score - 1e-15:
                best_score = sc
                best_h = h_trial.copy()
                h = h_trial.copy()
    
    return best_h, best_score

def construct():
    deadline = time.time() + 135.0  # safety margin
    
    # Phase 1: Sinusoidal optimization (structured low-dim search)
    h_sinusoidal, score_sinusoidal = _optimize_sinusoidal(deadline)
    
    # Phase 2: 2-block initialization
    h0 = _two_block_init()
    score_2block = _score(h0)
    
    # Choose best starting point
    if score_sinusoidal < score_2block:
        h_current = h_sinusoidal.copy()
        current_score = score_sinusoidal
    else:
        h_current = h0.copy()
        current_score = score_2block
    
    # Phase 3: SLSQP with moderate alpha
    try:
        if time.time() < deadline - 30:
            h_opt, new_score = _slsqp_solve(h_current, current_score + 0.05, 200.0, 1500, 1e-14, deadline)
            if h_opt is not None and new_score < current_score:
                h_current = h_opt.copy()
                current_score = new_score
    except Exception:
        pass
    
    # Phase 4: SLSQP with tighter alpha
    try:
        if time.time() < deadline - 25:
            h_opt, new_score = _slsqp_solve(h_current, current_score + 0.01, 500.0, 2000, 1e-16, deadline)
            if h_opt is not None and new_score < current_score:
                h_current = h_opt.copy()
                current_score = new_score
    except Exception:
        pass
    
    # Phase 5: Gradient descent refinement
    try:
        if time.time() < deadline - 10:
            h_opt, new_score = _gradient_descent(h_current, deadline)
            if new_score < current_score:
                h_current = h_opt.copy()
                current_score = new_score
    except Exception:
        pass
    
    # Phase 6: Final SLSQP polish
    try:
        if time.time() < deadline - 5:
            h_opt, new_score = _slsqp_solve(h_current, current_score + 0.005, 800.0, 1000, 1e-18, deadline)
            if h_opt is not None and new_score < current_score:
                h_current = h_opt.copy()
                current_score = new_score
    except Exception:
        pass
    
    # Final validation
    h_current = _project_to_feasible(h_current)
    final_score = _score(h_current)
    
    if not (np.all(h_current >= -1e-8) and np.all(h_current <= 1.0 + 1e-8)):
        h_current = np.full(N, 0.5)
    
    return h_current
