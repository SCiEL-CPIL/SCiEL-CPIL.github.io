"""Rewritten program: coupled variational optimization of log-energy on sphere."""
import numpy as np
from scipy.optimize import minimize
import math

N = 100


def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    d_sq = dists_sq[iu]
    d = np.sqrt(np.maximum(d_sq, 1e-30))
    return -np.sum(np.log(d))


def _log_energy_grad(points):
    """Vectorized gradient of the logarithmic energy with respect to points."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dists_sq = np.sum(diff * diff, axis=-1)  # (n, n)
    np.fill_diagonal(dists_sq, 1.0)
    
    # grad[k] = -sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # = -sum_{j != k} diff[k,j] / dists_sq[k,j]
    inv_d2 = np.zeros_like(dists_sq)
    np.fill_diagonal(inv_d2, 0.0)
    mask = dists_sq > 1e-30
    inv_d2[mask] = 1.0 / dists_sq[mask]
    
    # grad = -sum_j diff[k,j] * inv_d2[k,j] for each k
    # diff is (n,n,3), inv_d2 is (n,n)
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    return grad


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _constrained_energy_and_grad(x):
    """
    x: flattened array of 3*N coordinates.
    Returns energy and gradient for points on sphere.
    """
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)
    energy = _log_energy(points)
    
    grad_ambient = _log_energy_grad(points)
    
    # Project gradient onto tangent space of sphere
    radial = np.sum(grad_ambient * points, axis=1, keepdims=True)
    grad_tangent = grad_ambient - radial * points
    
    return energy, grad_tangent.ravel()


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    points0 = points0 + 0.01 * np.random.randn(N, 3)
    points0 = _project_to_sphere(points0)
    
    x0 = points0.ravel()
    
    best_points = points0.copy()
    best_energy = _log_energy(best_points)
    
    # Multi-start optimization with gradient descent + line search
    for restart in range(5):
        if restart == 0:
            pts = points0.copy()
        elif restart == 1:
            np.random.seed(100)
            pts = np.random.randn(N, 3)
            pts = _project_to_sphere(pts)
        elif restart == 2:
            np.random.seed(200)
            pts = points0 + 0.1 * np.random.randn(N, 3)
            pts = _project_to_sphere(pts)
        elif restart == 3:
            np.random.seed(300)
            pts = np.random.randn(N, 3)
            pts = _project_to_sphere(pts)
        else:
            np.random.seed(400)
            pts = best_points + 0.05 * np.random.randn(N, 3)
            pts = _project_to_sphere(pts)
        
        x_curr = pts.ravel()
        curr_energy = _log_energy(_project_to_sphere(x_curr.reshape(N, 3)))
        
        # Gradient descent with backtracking line search
        lr = 0.005
        for it in range(800):
            energy, grad_flat = _constrained_energy_and_grad(x_curr)
            grad = grad_flat.reshape(N, 3)
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < 1e-10:
                break
            
            # Backtracking line search
            alpha = lr
            found = False
            for _ in range(20):
                x_trial = x_curr - alpha * grad_flat
                points_trial = _project_to_sphere(x_trial.reshape(N, 3))
                trial_energy = _log_energy(points_trial)
                
                if trial_energy < energy:
                    x_curr = points_trial.ravel()
                    curr_energy = trial_energy
                    found = True
                    break
                alpha *= 0.5
            
            if not found:
                break
            
            # Adapt learning rate
            if trial_energy < energy - 0.001 * energy:
                lr = min(lr * 1.1, 0.05)
            else:
                lr = max(lr * 0.8, 0.0001)
            
            # Update best
            if curr_energy < best_energy:
                best_energy = curr_energy
                best_points = _project_to_sphere(x_curr.reshape(N, 3)).copy()
        
        # Update best from this restart
        final_energy = _log_energy(_project_to_sphere(x_curr.reshape(N, 3)))
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = _project_to_sphere(x_curr.reshape(N, 3)).copy()
    
    # Final polish with smaller steps
    x_final = best_points.ravel()
    lr = 0.001
    for it in range(300):
        energy, grad_flat = _constrained_energy_and_grad(x_final)
        grad = grad_flat.reshape(N, 3)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-11:
            break
        
        alpha = lr
        found = False
        for _ in range(15):
            x_trial = x_final - alpha * grad_flat
            points_trial = _project_to_sphere(x_trial.reshape(N, 3))
            trial_energy = _log_energy(points_trial)
            
            if trial_energy < energy:
                x_final = points_trial.ravel()
                found = True
                break
            alpha *= 0.5
        
        if not found:
            break
        
        if trial_energy < best_energy:
            best_energy = trial_energy
            best_points = _project_to_sphere(x_final.reshape(N, 3)).copy()
        
        lr = max(lr * 0.95, 0.00001)
    
    # Ensure valid output
    result = _project_to_sphere(best_points)
    
    # Validate: no points too close, all finite
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                result[j] += 1e-8 * np.random.randn(3)
                result[j] = result[j] / np.linalg.norm(result[j])
    
    if not np.all(np.isfinite(result)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        result = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return result
