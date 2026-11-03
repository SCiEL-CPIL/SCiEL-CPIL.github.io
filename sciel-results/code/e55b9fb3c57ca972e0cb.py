"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import time

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize


def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dists[i_idx, j_idx]
    valid = d_ij > 1e-12
    if not np.all(valid):
        d_ij = d_ij[valid]
    return -np.sum(np.log(d_ij))


def _energy_grad(points):
    """Compute energy and gradient."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dists[i_idx, j_idx]
    valid = d_ij > 1e-12
    d_ij = d_ij[valid]
    energy = -np.sum(np.log(d_ij))
    
    grad = np.zeros_like(points)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            v = points[i] - points[j]
            d2 = np.dot(v, v)
            if d2 < 1e-24:
                continue
            grad[i] += -v / d2
        
        # Project to tangent space: grad_i = (I - x_i x_i^T) grad_i
        grad[i] -= np.dot(grad[i], points[i]) * points[i]
    
    return energy, grad


def _fibonacci_sphere(n, offset=0.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def construct():
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin for safety
    
    # 1. Generate initial points using Fibonacci spiral
    pts = _fibonacci_sphere(N)
    
    best_pts = pts.copy()
    best_energy = _energy(best_pts)
    
    # 2. Optimize using L-BFGS-B with projected gradient
    def flat_to_points(flat):
        theta = flat[0:N]
        phi = flat[N:2*N]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
    
    def points_to_flat(pts):
        z = pts[:, 2]
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        return np.concatenate([theta, phi])
    
    def obj_grad(flat):
        pts = flat_to_points(flat)
        e, g_sphere = _energy_grad(pts)
        
        theta = flat[0:N]
        phi = flat[N:2*N]
        
        g_flat = np.zeros_like(flat)
        
        for i in range(N):
            t = theta[i]
            p = phi[i]
            
            g_flat[i] = (g_sphere[i, 0] * np.cos(t) * np.cos(p) +
                         g_sphere[i, 1] * np.cos(t) * np.sin(p) +
                         g_sphere[i, 2] * (-np.sin(t)))
            
            g_flat[i + N] = (g_sphere[i, 0] * (-np.sin(t) * np.sin(p)) +
                             g_sphere[i, 1] * (np.sin(t) * np.cos(p)))
        
        return e, g_flat
    
    # Initial flat parameters
    flat0 = points_to_flat(pts)
    
    stop_flag = [False]
    
    def callback(xk):
        if time.time() - start_time > deadline - 5.0:
            stop_flag[0] = True
            raise RuntimeError("Time limit reached")
        return stop_flag[0]
    
    try:
        res = minimize(obj_grad, flat0, jac=True, method='L-BFGS-B',
                       options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 100000},
                       callback=callback)
        if res.success or hasattr(res, 'x'):
            new_pts = flat_to_points(res.x)
            new_energy = _energy(new_pts)
            if new_energy < best_energy:
                best_pts = new_pts
                best_energy = new_energy
    except RuntimeError:
        pass
    except Exception as e:
        pass
    
    # 3. Gradient descent polish with adaptive step size
    if time.time() - start_time < deadline - 10.0:
        pts = best_pts.copy()
        step_size = 0.1
        lr = 0.01
        min_lr = 1e-6
        
        for it in range(5000):
            if time.time() - start_time > deadline - 5.0:
                break
            
            e, g = _energy_grad(pts)
            
            # Adaptive step size: reduce if energy increases
            pts_new = _project_to_sphere(pts - lr * g)
            e_new = _energy(pts_new)
            
            if e_new < e:
                pts = pts_new
                e = e_new
                # Increase learning rate slightly if improving
                lr = min(lr * 1.05, 0.1)
            else:
                # Reduce learning rate
                lr = max(lr * 0.5, min_lr)
                if lr <= min_lr * 1.1:
                    break
            
            if e < best_energy:
                best_energy = e
                best_pts = pts.copy()
        
        # Final high-precision polish with L-BFGS
        if time.time() - start_time < deadline - 5.0:
            flat_best = points_to_flat(best_pts)
            stop_flag[0] = False
            try:
                res2 = minimize(obj_grad, flat_best, jac=True, method='L-BFGS-B',
                                options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 100000},
                                callback=callback)
                if res2.success or hasattr(res2, 'x'):
                    new_pts = flat_to_points(res2.x)
                    new_energy = _energy(new_pts)
                    if new_energy < best_energy:
                        best_pts = new_pts
                        best_energy = new_energy
            except RuntimeError:
                pass
            except Exception as e:
                pass
    
    # Final check: ensure points are on the unit sphere and distinct
    norms = np.linalg.norm(best_pts, axis=1)
    best_pts = best_pts / norms[:, np.newaxis]
    
    return best_pts
# EVOLVE-BLOCK-END
