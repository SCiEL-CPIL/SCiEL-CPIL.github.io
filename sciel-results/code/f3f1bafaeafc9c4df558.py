"""Rewritten program: optimize 150 points on sphere for logarithmic energy."""
import numpy as np
import time
import math
from itertools import combinations

N = 150
TIME_LIMIT = 150.0  # seconds total
SAFETY_MARGIN = 10.0  # seconds to leave before hard limit

def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    # Compute pairwise distances
    # Use broadcasting for all pairs
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    # Mask diagonal and upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dists[mask]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0 / d)
    return math.fsum(terms.tolist())

def _gradient(pts):
    """Compute gradient of energy w.r.t. points (on sphere)."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d = np.sqrt(np.sum(diff * diff))
            if d < 1e-15:
                continue
            # d/dx_i log(1/|x_i - x_j|) = -1/|x_i - x_j| * (x_i - x_j)/|x_i - x_j|
            # = -(x_i - x_j) / |x_i - x_j|^2
            grad[i] -= diff / (d * d)
            grad[j] += diff / (d * d)
    # Project gradient onto tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    return grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.sqrt(np.sum(pts * pts, axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.sqrt(np.sum(pts * pts, axis=1, keepdims=True))
    return pts / norms

def _lbfgs_minimize(pts, max_iter=5000, tol=1e-10, time_start=None):
    """Minimize energy using L-BFGS with sphere constraint."""
    n = len(pts)
    pts = pts.copy()
    
    # Store history for L-BFGS
    m = 20  # memory
    s_hist = []  # steps
    y_hist = []  # gradients
    rho_hist = []
    
    step_size = 0.01
    current_energy = _energy(pts)
    current_grad = _gradient(pts)
    
    for it in range(max_iter):
        if time_start is not None:
            if time.time() - time_start > TIME_LIMIT - SAFETY_MARGIN:
                break
        
        # L-BFGS two-loop recursion to compute search direction
        if len(s_hist) > 0:
            q = current_grad.copy()
            alphas = []
            for k in range(len(s_hist) - 1, -1, -1):
                alpha = rho_hist[k] * np.sum(s_hist[k] * q)
                alphas.append(alpha)
                q -= alpha * y_hist[k]
            alphas.reverse()
            
            # Initial scaling
            if len(y_hist) > 0:
                gamma = np.sum(s_hist[-1] * y_hist[-1]) / np.sum(y_hist[-1] * y_hist[-1])
            else:
                gamma = 1.0
            r = gamma * q
            
            for k in range(len(s_hist)):
                beta = rho_hist[k] * np.sum(y_hist[k] * r)
                r += (alphas[k] - beta) * s_hist[k]
            direction = -r
        else:
            direction = -current_grad
        
        # Project direction to tangent space
        for i in range(n):
            direction[i] -= np.dot(direction[i], pts[i]) * pts[i]
        
        # Line search
        best_step = step_size
        best_energy = current_energy
        best_pts = pts.copy()
        
        for ls_iter in range(20):
            trial = pts + best_step * direction
            trial = _project_to_sphere(trial)
            trial_energy = _energy(trial)
            
            if trial_energy < best_energy:
                best_energy = trial_energy
                best_pts = trial.copy()
            else:
                break
            best_step *= 0.5
        
        if best_energy >= current_energy:
            # No improvement, reduce step size
            step_size *= 0.5
            if step_size < 1e-12:
                break
            continue
        
        # Update L-BFGS history
        s_new = best_pts - pts
        y_new = _gradient(best_pts) - current_grad
        sy = np.sum(s_new * y_new)
        
        if sy > 1e-12:
            s_hist.append(s_new)
            y_hist.append(y_new)
            rho_hist.append(1.0 / sy)
            if len(s_hist) > m:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
        
        current_pts = best_pts
        current_energy = best_energy
        current_grad = _gradient(current_pts)
        
        # Convergence check
        grad_norm = np.sqrt(np.sum(current_grad * current_grad))
        if grad_norm < tol:
            break
        
        # Reset step size if we made good progress
        if best_step == step_size:
            step_size = min(step_size * 1.1, 0.1)
    
    return _project_to_sphere(current_pts), current_energy

def _gradient_descent(pts, max_iter=10000, time_start=None):
    """Simple gradient descent with adaptive step size."""
    n = len(pts)
    pts = pts.copy()
    step_size = 0.005
    current_energy = _energy(pts)
    
    for it in range(max_iter):
        if time_start is not None:
            if time.time() - time_start > TIME_LIMIT - SAFETY_MARGIN:
                break
        
        grad = _gradient(pts)
        grad_norm = np.sqrt(np.sum(grad * grad))
        if grad_norm < 1e-12:
            break
        
        # Normalize gradient for consistent step
        direction = grad / (grad_norm + 1e-15)
        
        # Project to tangent space
        for i in range(n):
            direction[i] -= np.dot(direction[i], pts[i]) * pts[i]
        
        # Line search
        best_step = step_size
        best_energy = current_energy
        best_pts = pts.copy()
        
        for ls_iter in range(15):
            trial = pts + best_step * direction
            trial = _project_to_sphere(trial)
            trial_energy = _energy(trial)
            
            if trial_energy < best_energy:
                best_energy = trial_energy
                best_pts = trial.copy()
            else:
                break
            best_step *= 0.5
        
        if best_energy >= current_energy:
            step_size *= 0.5
            if step_size < 1e-14:
                break
            continue
        
        pts = best_pts
        current_energy = best_energy
        step_size = min(step_size * 1.05, 0.05)
    
    return _project_to_sphere(pts), current_energy

def _simulated_annealing(pts, time_start=None):
    """Simulated annealing to escape local minima."""
    n = len(pts)
    pts = pts.copy()
    current_energy = _energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    T = 0.1  # initial temperature
    cooling = 0.999
    perturb_scale = 0.01
    
    for it in range(5000):
        if time_start is not None:
            if time.time() - time_start > TIME_LIMIT - SAFETY_MARGIN:
                break
        
        # Perturb
        rng = np.random.default_rng(it)
        perturbation = rng.standard_normal((n, 3)) * perturb_scale
        trial = pts + perturbation
        trial = _project_to_sphere(trial)
        
        trial_energy = _energy(trial)
        delta = trial_energy - current_energy
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or np.random.random() < math.exp(-delta / (T + 1e-15)):
            pts = trial
            current_energy = trial_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        T *= cooling
    
    return _project_to_sphere(best_pts), best_energy

def construct():
    time_start = time.time()
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    current_energy = _energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    # First pass: L-BFGS optimization
    optimized_pts, opt_energy = _lbfgs_minimize(pts, max_iter=3000, time_start=time_start)
    if opt_energy < best_energy:
        best_energy = opt_energy
        best_pts = optimized_pts.copy()
    
    # Second pass: gradient descent from best
    gd_pts, gd_energy = _gradient_descent(best_pts, max_iter=5000, time_start=time_start)
    if gd_energy < best_energy:
        best_energy = gd_energy
        best_pts = gd_pts.copy()
    
    # Try a few random restarts if time permits
    for restart in range(5):
        if time.time() - time_start > TIME_LIMIT - SAFETY_MARGIN - 20:
            break
        
        random_pts = _random_points(N, seed=42 + restart)
        # Quick optimization
        rp_opt, rp_energy = _lbfgs_minimize(random_pts, max_iter=1000, time_start=time_start)
        if rp_energy < best_energy:
            best_energy = rp_energy
            best_pts = rp_opt.copy()
    
    # Final polish with gradient descent
    final_pts, final_energy = _gradient_descent(best_pts, max_iter=2000, time_start=time_start)
    if final_energy < best_energy:
        best_energy = final_energy
        best_pts = final_pts.copy()
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((best_pts[i] - best_pts[j]) ** 2))
            min_dist = min(min_dist, d)
    
    if min_dist < 1e-9:
        # Fall back to Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
    
    return best_pts

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
