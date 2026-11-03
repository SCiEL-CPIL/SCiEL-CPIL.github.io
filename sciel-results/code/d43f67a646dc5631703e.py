import numpy as np
import time
import math

N = 100

def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|)"""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    return float(-np.sum(np.log(dist[iu])))

def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = diff / dist_sq[:, :, np.newaxis]
    idx = np.diag_indices(N)
    terms[idx] = 0.0
    grad = -np.sum(terms, axis=1)
    
    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1)[:, np.newaxis]
    grad_proj = grad - dot * pts
    
    # Energy
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    return energy, grad_proj

def _fibonacci_spiral(n, offset=0.0, jitter=0.0):
    """Generate points on sphere using Fibonacci spiral with optional offset and jitter."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        x += jitter * np.random.randn(n, 3)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms
    return x

def _project(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _gradient_descent(pts, max_iters=500, lr=0.01, decay=0.999, deadline=None):
    """Riemannian gradient descent with adaptive learning rate."""
    best_x = pts.copy()
    best_energy = _energy(pts)
    
    for t in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_x = pts.copy()
        
        # Backtracking line search
        step = lr
        for _ in range(10):
            new_pts = _project(pts - step * grad)
            new_energy = _energy(new_pts)
            if new_energy < energy:
                break
            step *= 0.5
        
        pts = new_pts
        lr *= decay
    
    return best_x, best_energy

def _scipy_l_bfgs(pts, deadline=None):
    """Use scipy L-BFGS-B for final optimization."""
    from scipy.optimize import minimize
    
    def objective_and_grad(flat):
        pts = _project(flat.reshape(N, 3))
        energy, grad = _energy_grad(pts)
        return energy, grad.flatten()
    
    x0 = pts.flatten()
    
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 100, 'gtol': 1e-8, 'disp': False}
    )
    
    if result.success or result.fun < _energy(pts):
        return _project(result.x.reshape(N, 3)), float(result.fun)
    
    return pts, _energy(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Try multiple restarts with different initializations
    best_x = None
    best_energy = float('inf')
    
    # Phase 1: Gradient descent from multiple initializations
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 30:  # Leave 30s for L-BFGS
            break
        
        # Different initializations
        if r == 0:
            pts = _fibonacci_spiral(N)
        elif r == 1:
            pts = _fibonacci_spiral(N, offset=0.5)
        elif r == 2:
            pts = _fibonacci_spiral(N, offset=1.0)
        elif r == 3:
            pts = _fibonacci_spiral(N, jitter=0.05)
        else:
            pts = np.random.randn(N, 3)
            pts = _project(pts)
        
        # Gradient descent
        x_opt, e_opt = _gradient_descent(
            pts, 
            max_iters=300, 
            lr=0.01, 
            decay=0.999,
            deadline=deadline - 30
        )
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt.copy()
    
    # Phase 2: L-BFGS polish on the best solution
    if best_x is not None and time.time() < deadline - 5:
        x_polished, e_polished = _scipy_l_bfgs(best_x, deadline=deadline - 2)
        if e_polished < best_energy:
            best_energy = e_polished
            best_x = x_polished
    
    # Final check: ensure valid output
    best_x = _project(best_x)
    assert best_x.shape == (N, 3)
    assert np.all(np.isfinite(best_x))
    assert np.all(np.linalg.norm(best_x, axis=1) > 1e-9)
    
    return best_x
