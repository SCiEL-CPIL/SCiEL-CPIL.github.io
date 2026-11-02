"""Rewritten program: coupled variational optimization of log-energy on sphere."""
import numpy as np
from scipy.optimize import minimize
import math

N = 100


def _log_energy_vec(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|) vectorized."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(sq_dists, 1.0)
    iu = np.triu_indices(N, k=1)
    d_sq = sq_dists[iu]
    d_sq = np.maximum(d_sq, 1e-15)
    return -0.5 * np.sum(np.log(d_sq))


def _log_energy_grad_vec(points):
    """Gradient of the logarithmic energy with respect to points, vectorized."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(sq_dists, 1.0)
    inv_dists_sq = 1.0 / np.maximum(sq_dists, 1e-15)
    np.fill_diagonal(inv_dists_sq, 0.0)
    sum_inv = np.sum(inv_dists_sq, axis=1, keepdims=True)
    weighted_sum = inv_dists_sq @ points
    grad = - (points * sum_inv - weighted_sum)
    return grad


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    inits = [fib_points]
    for seed in range(5):
        np.random.seed(42 + seed * 100)
        pts = np.random.randn(N, 3)
        inits.append(_project_to_sphere(pts))
    np.random.seed(123)
    inits.append(_project_to_sphere(fib_points + 0.1 * np.random.randn(N, 3)))
    
    best_points = None
    best_energy = np.inf
    
    def obj_and_grad(x):
        pts = x.reshape(N, 3)
        e = _log_energy_vec(pts)
        g = _log_energy_grad_vec(pts)
        return e, g.ravel()
        
    def constraint(x):
        pts = x.reshape(N, 3)
        c = np.sum(pts * pts, axis=1) - 1.0
        jac = 2.0 * pts
        return c, jac

    for init in inits:
        x0 = init.ravel()
        try:
            res = minimize(
                obj_and_grad,
                x0,
                method='SLSQP',
                jac=True,
                constraints=[{'type': 'eq', 'fun': constraint}],
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            if res.success:
                pts = res.x.reshape(N, 3)
                pts = _project_to_sphere(pts)
                e = _log_energy_vec(pts)
                if e < best_energy:
                    best_energy = e
                    best_points = pts
        except Exception:
            pass
            
    if best_points is None:
        best_points = fib_points
        best_energy = _log_energy_vec(best_points)
        
    x0 = best_points.ravel()
    try:
        res = minimize(
            obj_and_grad,
            x0,
            method='SLSQP',
            jac=True,
            constraints=[{'type': 'eq', 'fun': constraint}],
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        if res.success:
            pts = res.x.reshape(N, 3)
            pts = _project_to_sphere(pts)
            e = _log_energy_vec(pts)
            if e < best_energy:
                best_energy = e
                best_points = pts
    except Exception:
        pass
        
    result = _project_to_sphere(best_points)
    
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                result[j] += 1e-8 * np.random.randn(3)
                result[j] = result[j] / np.linalg.norm(result[j])
                
    if not np.all(np.isfinite(result)):
        result = fib_points
        
    return result
