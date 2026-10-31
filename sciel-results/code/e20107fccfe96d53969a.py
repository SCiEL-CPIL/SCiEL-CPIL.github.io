"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    if n == 0:
        return 0.0
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _project_to_feasible(h):
    """Project h to satisfy sum=N/2 and 0<=h<=1."""
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s < 1e-10:
        h = np.full(N, 0.5)
        return h
    h = h * (TARGET_SUM / s)
    h = np.clip(h, 0.0, 1.0)
    # Fix residual iteratively
    for _ in range(20):
        diff = TARGET_SUM - h.sum()
        if abs(diff) < 1e-15:
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

def _objective_smooth(params, alpha=200.0):
    """Objective with epigraph: minimize t + penalty for violation.
    params = [h_0, ..., h_{N-1}, t]
    """
    h = params[:N]
    t = params[N]
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    smax = _smooth_max(overlap, alpha)
    penalty = 1000.0 * np.maximum(0.0, smax - t)**2
    return t + penalty

def _objective_smooth_grad(params, alpha=200.0):
    """Objective with analytic gradient."""
    h = params[:N]
    t = params[N]
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    smax = _smooth_max(overlap, alpha)
    penalty = 1000.0 * np.maximum(0.0, smax - t)**2
    val = t + penalty
    
    # Gradient
    grad = np.zeros(N + 1)
    # d/dt
    if smax > t:
        grad[N] = 1.0 - 2000.0 * (smax - t)
    else:
        grad[N] = 1.0
    
    # d/dh_i for i in 0..N-1
    # overlap_i = h_i * (1 - h_{i+1}) for i < N-1
    # overlap_{N-1} = h_{N-1} * (1 - 0) = h_{N-1}
    
    w = _smooth_max_grad(overlap, alpha)
    
    # d(smax)/d(overlap_k) = w[k]
    
    # d(overlap_i)/d(h_j) matrix is sparse
    for i in range(N - 1):
        if w[i] > 1e-15:
            grad[i] += w[i] * (1.0 - h[i+1])
            grad[i+1] += w[i] * (-h[i])
    # i = N-1: overlap = h[N-1], d/dh[N-1] = 1
    if w[N-1] > 1e-15:
        grad[N-1] += w[N-1] * 1.0
    
    # Penalty gradient
    if smax > t:
        dp = 2000.0 * (smax - t)
        # d(penalty)/d(h_j) = dp * d(smax)/d(h_j)
        grad_smax_h = np.zeros(N)
        for i in range(N - 1):
            if w[i] > 1e-15:
                grad_smax_h[i] += w[i] * (1.0 - h[i+1])
                grad_smax_h[i+1] += w[i] * (-h[i])
        if w[N-1] > 1e-15:
            grad_smax_h[N-1] += w[N-1]
        
        grad[:N] += dp * grad_smax_h
        grad[N] -= dp  # d(penalty)/dt = -2000*(smax-t) when smax>t
    
    return val, grad

def _two_block_init():
    """Analytic 2-block initial guess."""
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382
    B = 1.0 - A  # ~0.618
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    return h0

def _generate_knots_init(n_knots=10):
    """Generate initial knot positions and values for piecewise linear representation."""
    # Knot positions in [0, 1]
    positions = np.linspace(0, 1, n_knots)
    # Values in [0, 1]
    values = np.full(n_knots, 0.5)
    return positions, values

def _knots_to_h(positions, values):
    """Interpolate knots to get N step heights."""
    h = np.interp(np.linspace(0, 1, N), positions, values)
    return h

def _construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Initial best from analytic 2-block solution
    h0 = _two_block_init()
    h_final = h0.copy()
    current_score = _score(h0)
    
    # Phase 1: Global Search using Differential Evolution on full h vector
    try:
        if time.time() < deadline - 25.0:
            def de_objective(h_params):
                h = _project_to_feasible(h_params)
                return _score(h)
            
            bounds = [(0.0, 1.0)] * N
            
            result_de = minimize(
                de_objective,
                h0,
                method='DifferentialEvolution',
                bounds=bounds,
                options={
                    'maxiter': 80, 
                    'popsize': 25, 
                    'tol': 1e-12,
                    'seed': 42,
                    'polish': False
                }
            )
            
            if result_de.fun < current_score:
                h_de = _project_to_feasible(result_de.x)
                score_de = _score(h_de)
                if score_de < current_score:
                    h_final = h_de.copy()
                    current_score = score_de
    except Exception:
        pass
    
    # Phase 2: Knot-based local optimization using Nelder-Mead
    # Represent h as piecewise-linear interpolation of knots
    # This captures sharp transitions more efficiently
    try:
        if time.time() < deadline - 15.0:
            n_knots = 12  # Number of knots for piecewise linear
            
            # Initialize knots from current best h
            # Find local extrema and significant changes to place knots
            h_init = h_final.copy()
            
            # Simple knot placement: evenly spaced with some jitter
            positions = np.linspace(0, 1, n_knots)
            # Add small random perturbations to positions (except endpoints)
            for i in range(1, n_knots - 1):
                positions[i] += random.uniform(-0.02, 0.02)
            positions = np.sort(positions)
            positions[0] = 0.0
            positions[-1] = 1.0
            
            # Interpolate initial values from h_init
            values = np.interp(positions, np.linspace(0, 1, N), h_init)
            
            # Objective function for knot optimization
            def knot_objective(params):
                pos = params[:n_knots]
                val = params[n_knots:]
                
                # Ensure positions are sorted and within [0, 1]
                pos = np.sort(np.clip(pos, 0, 1))
                pos[0] = 0.0
                pos[-1] = 1.0
                
                # Clip values to [0, 1]
                val = np.clip(val, 0, 1)
                
                # Interpolate to get h
                h = np.interp(np.linspace(0, 1, N), pos, val)
                
                # Project to feasible set
                h = _project_to_feasible(h)
                
                return _score(h)
            
            # Initial parameters
            x0_knots = np.concatenate([positions, values])
            
            # Bounds for Nelder-Mead (approximate, since it doesn't use bounds directly)
            # We'll handle bounds via clipping in the objective
            
            result_knot = minimize(
                knot_objective,
                x0_knots,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
            )
            
            if result_knot.fun < current_score:
                pos_opt = np.sort(np.clip(result_knot.x[:n_knots], 0, 1))
                pos_opt[0] = 0.0
                pos_opt[-1] = 1.0
                val_opt = np.clip(result_knot.x[n_knots:], 0, 1)
                
                h_knot = np.interp(np.linspace(0, 1, N), pos_opt, val_opt)
                h_knot = _project_to_feasible(h_knot)
                score_knot = _score(h_knot)
                
                if score_knot < current_score:
                    h_final = h_knot.copy()
                    current_score = score_knot
    except Exception:
        pass
    
    # Phase 3: Local Polish using SLSQP with epigraph formulation
    try:
        if time.time() < deadline - 5.0:
            def objective_polish(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=500.0)
                return t + 1000.0 * np.maximum(0.0, smax - t)**2
            
            x0_polish = np.concatenate([h_final, [current_score + 0.01]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def constraint_sum(params):
                return np.sum(params[:N]) - TARGET_SUM
            
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
            
            result2 = minimize(
                objective_polish,
                x0_polish,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            
            if result2.fun < 100.0:
                h_opt2 = result2.x[:N].copy()
                h_opt2 = _project_to_feasible(h_opt2)
                new_score2 = _score(h_opt2)
                if new_score2 < current_score:
                    h_final = h_opt2.copy()
                    current_score = new_score2
    except Exception:
        pass
    
    # Phase 4: Final high-precision polish with L-BFGS-B on smooth objective
    try:
        if time.time() < deadline - 2.0:
            def obj_lbfgs(h):
                h = np.clip(h, 0, 1)
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return _smooth_max(overlap, alpha=1000.0)
            
            def grad_lbfgs(h):
                h = np.clip(h, 0, 1)
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                w = _smooth_max_grad(overlap, alpha=1000.0)
                
                grad = np.zeros(N)
                for i in range(N - 1):
                    if w[i] > 1e-15:
                        grad[i] += w[i] * (1.0 - h[i+1])
                        grad[i+1] += w[i] * (-h[i])
                if w[N-1] > 1e-15:
                    grad[N-1] += w[N-1]
                
                return grad
            
            result3 = minimize(
                obj_lbfgs,
                h_final,
                jac=grad_lbfgs,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'iprint': 0}
            )
            
            if result3.fun < current_score + 0.01:
                h_opt3 = _project_to_feasible(result3.x)
                new_score3 = _score(h_opt3)
                if new_score3 < current_score:
                    h_final = h_opt3.copy()
                    current_score = new_score3
    except Exception:
        pass
    
    # Final validation
    h_final = _project_to_feasible(h_final)
    final_score = _score(h_final)
    
    if not (np.all(h_final >= -1e-8) and np.all(h_final <= 1.0 + 1e-8)):
        h_final = np.full(N, 0.5)
    
    return h_final

def construct():
    return _construct()
