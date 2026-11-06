"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """Compute energy and gradient (projected to tangent space)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal and handle small distances
    mask = ~np.eye(N, dtype=bool)
    dists_safe = np.where(dists < 1e-12, 1e-12, dists)
    
    # Energy
    d = dists_safe[mask]
    energy = math.fsum(np.log(1.0 / d).tolist())
    
    # Gradient
    W = np.zeros((N, N))
    W[mask] = 1.0 / (dists_safe[mask] ** 2)
    
    sum_W = W.sum(axis=1, keepdims=True)
    grad = W @ pts - sum_W * pts
    
    # Project to tangent space
    dot_prod = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot_prod * pts
    
    return energy, grad

def _momentum_descent(x0, max_iter=10000, lr=0.01, momentum=0.9, time_limit=10.0):
    """Run momentum gradient descent."""
    pts = x0.reshape(-1, 3).copy()
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms
    
    v = np.zeros_like(pts)
    start_time = time.time()
    best_energy = float('inf')
    best_pts = pts.copy()
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        energy, grad = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Update velocity
        v = momentum * v - lr * grad
        
        # Update position
        pts = pts + v
        
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts = pts / norms
        
    return best_pts, best_energy

def _log_energy_flat(x):
    """Compute logarithmic energy from flat vector for L-BFGS-B polish."""
    pts = x.reshape(-1, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_upper, j_upper = np.triu_indices(N, k=1)
    d = dists[i_upper, j_upper]
    d = np.maximum(d, 1e-12)
    
    terms = np.log(1.0 / d)
    return math.fsum(terms.tolist())

def _gradient_flat(x):
    """Compute gradient from flat vector for L-BFGS-B polish."""
    pts = x.reshape(-1, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    dists_safe = np.where(dists < 1e-12, 1e-12, dists)
    
    W = np.zeros((N, N))
    mask = ~np.eye(N, dtype=bool)
    W[mask] = 1.0 / (dists_safe[mask] ** 2)
    
    sum_W = W.sum(axis=1, keepdims=True)
    grad = W @ pts - sum_W * pts
    
    # Project to tangent space
    dot_prod = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot_prod * pts
    
    return grad.flatten()

def construct():
    start_time = time.time()
    max_time = 115.0
    
    # Generate multiple starting points
    starts = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts1 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts1, axis=1, keepdims=True)
    pts1 = pts1 / norms
    starts.append(pts1.flatten())
    
    # 2. Random Uniform
    np.random.seed(42)
    pts2 = np.random.randn(N, 3)
    norms = np.linalg.norm(pts2, axis=1, keepdims=True)
    pts2 = pts2 / norms
    starts.append(pts2.flatten())
    
    # 3. Halton Sequence
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    pts3 = np.zeros((N, 3))
    for i in range(N):
        pts3[i, 0] = 2 * halton(i+1, 2) - 1
        pts3[i, 1] = 2 * halton(i+1, 3) - 1
        pts3[i, 2] = 2 * halton(i+1, 5) - 1
        norm = np.linalg.norm(pts3[i])
        pts3[i] /= norm
    starts.append(pts3.flatten())
    
    # 4. Another random seed
    np.random.seed(123)
    pts4 = np.random.randn(N, 3)
    norms = np.linalg.norm(pts4, axis=1, keepdims=True)
    pts4 = pts4 / norms
    starts.append(pts4.flatten())

    best_x = None
    best_energy = float('inf')
    
    # Try different hyperparameters for Momentum Descent
    params = [
        (0.05, 0.9),
        (0.02, 0.9),
        (0.1, 0.8),
        (0.01, 0.95)
    ]
    
    for x0 in starts:
        if time.time() - start_time > max_time - 5.0:
            break
            
        for lr, mom in params:
            if time.time() - start_time > max_time - 5.0:
                break
            try:
                # Allocate time based on remaining time
                time_left = max_time - (time.time() - start_time)
                t_limit = min(10.0, time_left / 2.0) 
                if t_limit < 1.0:
                    continue
                    
                x_opt, energy = _momentum_descent(x0, max_iter=20000, lr=lr, momentum=mom, time_limit=t_limit)
                if energy < best_energy:
                    best_energy = energy
                    best_x = x_opt
            except Exception:
                continue

    if best_x is None:
        best_x = starts[0]
        
    # Final Polish with L-BFGS-B
    if time.time() - start_time < max_time - 2.0:
        try:
            res = minimize(
                _log_energy_flat,
                best_x,
                jac=_gradient_flat,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_energy:
                best_x = res.x
                best_energy = res.fun
        except Exception:
            pass

    pts = best_x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms
    
    return pts
# EVOLVE-BLOCK-END
