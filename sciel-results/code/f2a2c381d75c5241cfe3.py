"""Two-stage optimization: repulsive spread + smooth covering-radius refinement."""
import numpy as np
from scipy.optimize import minimize
import time

N = 130
_DEADLINE = 150.0
_START_TIME = time.time()

def _remaining():
    return _DEADLINE - 5.0 - (time.time() - _START_TIME)  # 5s safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _soft_max_covering_radius(pts, beta=50.0):
    """
    Smooth approximation of the covering radius.
    For each point on a fine grid of the sphere, compute min angular distance to sites,
    then take a soft-max (log-sum-exp) over the grid to approximate the max.
    Actually, better: use the convex hull facet circumradius approach is hard to differentiate.
    Instead, we use a grid-based soft-max.
    """
    pts = _normalize(pts)
    # Create a reasonably fine grid on the sphere
    # Use Fibonacci sphere for grid points
    n_grid = 2000
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distances: cos(angle) = dot product
    # For each grid point, min distance to any site
    # dot: (n_grid, N)
    dots = grid @ pts.T  # (n_grid, N)
    dots = np.clip(dots, -1.0, 1.0)
    # angular distance = arccos(dot)
    # We want min over sites of arccos(dot), then max over grid
    # Equivalent: max over grid of min over sites of arccos(dot)
    # = max over grid of arccos(max over sites of dot)
    max_dots = np.max(dots, axis=1)  # (n_grid,)
    angles = np.arccos(max_dots)  # (n_grid,)
    
    # Soft-max of angles (approximate max)
    # log-sum-exp for stability
    m = np.max(angles)
    if m < 1e-15:
        return m
    log_sum = m + np.log(np.sum(np.exp(angles - m)))
    # This gives a soft-max; for large beta it approximates max
    # Actually, log-sum-exp of angles isn't quite right. Let's use:
    # soft_max = (1/beta) * log(sum(exp(beta * angles)))
    # But angles are in [0, pi], so beta*angles can be large
    # Let's just return the actual max for simplicity, or use a smooth approximation
    # For differentiability, use:
    # soft_max = m + (1/beta) * log(sum(exp(beta*(angles - m))))
    soft_max_val = m + (1.0 / beta) * np.log(np.sum(np.exp(beta * (angles - m))))
    return soft_max_val

def _repulsive_potential(pts, power=4.0, scale=1.0):
    """
    Sum of inverse-power distances to prevent coincident points.
    """
    pts = _normalize(pts)
    # Pairwise dot products
    dots = pts @ pts.T
    np.fill_diagonal(dots, 0.0)
    # angular distance = arccos(dot), but for repulsion we want large when close
    # Use 1 / (1 - dot)^power or similar
    # 1 - dot is proportional to chord distance squared
    # Let's use: sum over pairs of 1 / (1 - dot)^power
    # But this can blow up. Use a smoothed version.
    one_minus_dot = 1.0 - dots
    one_minus_dot = np.maximum(one_minus_dot, 1e-6)
    repulsive = np.sum(one_minus_dot ** (-power))
    return repulsive

def _objective(x, beta=50.0, rep_weight=0.01, power=4.0):
    """
    Combined objective: covering radius (soft-max) + repulsive term.
    """
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    cov = _soft_max_covering_radius(pts, beta=beta)
    rep = _repulsive_potential(pts, power=power)
    return cov + rep_weight * rep

def _gradient(x, beta=50.0, rep_weight=0.01, power=4.0):
    """
    Numerical gradient (central difference) for simplicity.
    """
    eps = 1e-6
    grad = np.zeros_like(x)
    f0 = _objective(x, beta, rep_weight, power)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (_objective(x_plus, beta, rep_weight, power) - _objective(x_minus, beta, rep_weight, power)) / (2 * eps)
    return grad

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Stage 1: Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    x0 = _normalize(x0)
    
    best_x = x0.copy()
    best_obj = _objective(x0, beta=50.0, rep_weight=0.01, power=4.0)
    
    # Stage 2: L-BFGS-B optimization
    # Use a moderate beta for smoothness during optimization
    beta = 30.0
    rep_weight = 0.005
    power = 4.0
    
    try:
        result = minimize(
            _objective,
            x0,
            args=(beta, rep_weight, power),
            method='L-BFGS-B',
            jac=_gradient,
            options={
                'maxiter': 200,
                'ftol': 1e-10,
                'gtol': 1e-8,
                'maxfun': 5000,
                'disp': False,
            }
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            obj_val = _objective(result.x, beta=beta, rep_weight=rep_weight, power=power)
            if obj_val < best_obj:
                best_x = result.x.copy()
                best_obj = obj_val
    except Exception:
        pass
    
    # Check deadline
    if _remaining() < 20.0:
        return _normalize(best_x)
    
    # Stage 3: Polish with higher beta (closer to true max)
    beta2 = 100.0
    rep_weight2 = 0.001
    x_start = _normalize(best_x)
    best_obj2 = _objective(x_start, beta=beta2, rep_weight=rep_weight2, power=power)
    
    try:
        result2 = minimize(
            _objective,
            x_start,
            args=(beta2, rep_weight2, power),
            method='L-BFGS-B',
            jac=_gradient,
            options={
                'maxiter': 100,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'maxfun': 3000,
                'disp': False,
            }
        )
        if result2.x is not None and np.all(np.isfinite(result2.x)):
            obj_val2 = _objective(result2.x, beta=beta2, rep_weight=rep_weight2, power=power)
            if obj_val2 < best_obj2:
                best_x = result2.x.copy()
                best_obj2 = obj_val2
    except Exception:
        pass
    
    # Check deadline
    if _remaining() < 10.0:
        return _normalize(best_x)
    
    # Stage 4: Final polish with very high beta
    beta3 = 500.0
    rep_weight3 = 0.0
    x_start3 = _normalize(best_x)
    best_obj3 = _objective(x_start3, beta=beta3, rep_weight=rep_weight3, power=power)
    
    try:
        result3 = minimize(
            _objective,
            x_start3,
            args=(beta3, rep_weight3, power),
            method='L-BFGS-B',
            jac=_gradient,
            options={
                'maxiter': 50,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxfun': 2000,
                'disp': False,
            }
        )
        if result3.x is not None and np.all(np.isfinite(result3.x)):
            obj_val3 = _objective(result3.x, beta=beta3, rep_weight=rep_weight3, power=power)
            if obj_val3 < best_obj3:
                best_x = result3.x.copy()
    except Exception:
        pass
    
    return _normalize(best_x)
