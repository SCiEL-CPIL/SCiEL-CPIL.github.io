import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 150.0
DEADLINE_FRACTION = 0.92

def _log_energy_fast(pts):
    """Vectorized log energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, 1.0)
    iu = np.triu_indices(n, k=1)
    return -0.5 * np.sum(np.log(dists2[iu]))

def _log_energy_grad_fast(pts):
    """Vectorized log energy and gradient."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, 1.0)
    
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dists2[iu]))
    
    inv_dists2 = 1.0 / dists2
    np.fill_diagonal(inv_dists2, 0.0)
    
    s = (diff * inv_dists2[:, :, np.newaxis]).sum(axis=1)
    grad = -s
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _tangent_project(pts, grad):
    """Project gradient to tangent space of sphere."""
    coeffs = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - coeffs * pts

def _fibonacci_spiral(n=N):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n=N, seed=None):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)

def _optimize_lbfgs(pts, deadline, max_iter=10000):
    """Run L-BFGS-B optimization with projected gradients."""
    pts = _project_to_sphere(pts)
    x0 = pts.flatten()
    
    def func(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _log_energy_grad_fast(p)
        return e
        
    def jac(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _log_energy_grad_fast(p)
        g_t = _tangent_project(p, g)
        return g_t.flatten()
        
    options = {
        'maxiter': max_iter,
        'gtol': 1e-14,
        'ftol': 1e-18,
        'disp': False,
        'maxfun': 50000,
        'maxls': 50,
    }
    
    try:
        result = minimize(func, x0, jac=jac, method='L-BFGS-B', options=options)
        if result.fun < 1e17 and np.all(np.isfinite(result.x)):
            return _project_to_sphere(result.x.reshape(-1, 3))
    except Exception:
        pass
        
    return pts

def _polish(pts, deadline, initial_lr=0.01, min_lr=1e-15, max_iterations=5000):
    """High-precision steepest descent polish with backtracking line search."""
    pts = _project_to_sphere(pts)
    current_energy = _log_energy_fast(pts)
    
    lr = initial_lr
    patience = 0
    max_patience = 100
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
            
        e, g = _log_energy_grad_fast(pts)
        if e < -1e17:
            break
        
        g_t = _tangent_project(pts, g)
        grad_norm = np.linalg.norm(g_t)
        
        if grad_norm < 1e-16:
            break
        
        alpha = lr
        found = False
        for _ in range(50):
            new_pts = pts - alpha * g_t
            new_pts = _project_to_sphere(new_pts)
            
            new_e = _log_energy_fast(new_pts)
            
            if new_e < current_energy - 1e-18 * alpha * grad_norm:
                pts = new_pts
                current_energy = new_e
                found = True
                break
            alpha *= 0.5
        
        if found:
            patience = 0
            lr = min(alpha * 2.0, 0.1)
        else:
            lr = alpha * 0.5
            patience += 1
            if patience > max_patience or lr < min_lr:
                break
                
    return pts

def _newton_polish(pts, deadline):
    """Final ultra-precise polish with very small steps."""
    pts = _project_to_sphere(pts)
    current_energy = _log_energy_fast(pts)
    
    for step in range(1000):
        if time.time() > deadline:
            break
            
        e, g = _log_energy_grad_fast(pts)
        g_t = _tangent_project(pts, g)
        grad_norm = np.linalg.norm(g_t)
        
        if grad_norm < 1e-18:
            break
        
        alpha = 0.001
        for _ in range(60):
            new_pts = pts - alpha * g_t
            new_pts = _project_to_sphere(new_pts)
            new_e = _log_energy_fast(new_pts)
            
            if new_e < current_energy - 1e-20 * alpha * grad_norm:
                pts = new_pts
                current_energy = new_e
                break
            alpha *= 0.5
        else:
            if alpha < 1e-20:
                break
                
    return pts

def _riemannian_flow(pts, time_budget, initial_lr=0.05, patience=1000):
    """Riemannian gradient descent with backtracking line search."""
    start_time = time.time()
    best_points = pts.copy()
    best_energy, _ = _log_energy_grad_fast(best_points)
    lr = initial_lr
    no_improve = 0
    
    while time.time() - start_time < time_budget:
        energy, grad = _log_energy_grad_fast(best_points)
        grad_t = _tangent_project(best_points, grad)
        grad_norm = np.linalg.norm(grad_t)
        
        if grad_norm < 1e-12:
            break
        
        alpha = lr
        improved = False
        for _ in range(30):
            new_points = best_points - alpha * grad_t
            new_points = _project_to_sphere(new_points)
            
            new_energy, _ = _log_energy_grad_fast(new_points)
            
            if new_energy < energy - 1e-12:
                best_points = new_points
                best_energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if improved:
            no_improve = 0
            lr = min(alpha * 1.5, 0.5)
        else:
            no_improve += 1
            lr *= 0.9
            if no_improve > patience:
                break
    
    return best_points

def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + TIME_LIMIT * DEADLINE_FRACTION

    # Start with Fibonacci spiral - best known configuration
    best_pts = _fibonacci_spiral(N)
    best_energy = _log_energy_fast(best_pts)
    
    # Phase 1: Quick Riemannian flow to get into good basin
    flow_deadline = _start_time + (deadline - _start_time) * 0.25
    best_pts = _riemannian_flow(best_pts, flow_deadline - _start_time, initial_lr=0.1, patience=500)
    best_energy = _log_energy_fast(best_pts)
    
    # Phase 2: L-BFGS-B optimization - main workhorse
    lbfgs_deadline = _start_time + (deadline - _start_time) * 0.70
    time_for_lbfgs = lbfgs_deadline - time.time()
    if time_for_lbfgs > 5.0:
        # Estimate iterations based on available time
        max_iter = min(20000, int(time_for_lbfgs * 500))
        best_pts = _optimize_lbfgs(best_pts, lbfgs_deadline, max_iter=max_iter)
        best_energy = _log_energy_fast(best_pts)
        
        # Quick polish after L-BFGS
        if time.time() < lbfgs_deadline - 2.0:
            polished = _polish(best_pts, lbfgs_deadline, initial_lr=0.01)
            pe = _log_energy_fast(polished)
            if pe < best_energy:
                best_energy = pe
                best_pts = polished
    
    # Phase 3: Final high-precision polish with remaining time
    polish_deadline = deadline - 2.0
    if time.time() < polish_deadline:
        best_pts = _polish(best_pts, polish_deadline, initial_lr=0.005, min_lr=1e-16)
        best_energy = _log_energy_fast(best_pts)
        
        # Ultra-precise Newton-style polish
        if time.time() < polish_deadline - 1.0:
            best_pts = _newton_polish(best_pts, polish_deadline)
            best_energy = _log_energy_fast(best_pts)
    
    # Final projection and validity check
    best_pts = _project_to_sphere(best_pts)
    
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_spiral(N)
        
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    if min_d < 1e-9:
        return _fibonacci_spiral(N)
        
    return best_pts
