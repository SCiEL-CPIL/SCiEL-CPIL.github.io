"""
Strategy: Deterministic L-BFGS-B optimization on the sphere using vectorized
energy/gradient computations, seeded from diverse initializations.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
import math

def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    and its gradient w.r.t. points on the sphere (tangent space).
    
    E = -sum_{i<j} log(r_ij) where r_ij = |xi - xj|
    dE/dxi = -sum_{j != i} d(-log(r_ij))/dxi = -sum_{j != i} (-(xi - xj)/r_ij^2) * 1/r_ij * (-1) ... wait.
    
    Let f = -log(r). df/dr = -1/r.
    dr/dxi = (xi - xj)/r.
    df/dxi = -1/r * (xi - xj)/r = -(xi - xj)/r^2.
    So dE/dxi = sum_{j != i} -(xi - xj)/r_ij^2 = sum_{j != i} (xj - xi)/r_ij^2.
    
    Wait, the energy is sum log(1/d) = -sum log(d).
    d/dxi (-log|xi-xj|) = - (1/|xi-xj|) * d|xi-xj|/dxi = - (1/r) * (xi-xj)/r = -(xi-xj)/r^2.
    So gradient is sum_{j!=i} -(xi-xj)/r_ij^2.
    
    This is the gradient in R^3. We must project to the tangent space.
    """
    # Compute pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :] # (N, N, 3)
    # Compute distances
    dists = np.linalg.norm(diff, axis=2) # (N, N)
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-12, 1e-12, dists)
    
    # Energy: sum over i < j of -log(dists[i,j])
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    energy = -np.sum(np.log(pair_dists))
    
    # Gradient: grad_i = sum_{j != i} -(xi - xj) / r_ij^2
    # Note: diff[i,j] = xi - xj.
    # We need to sum over j for each i.
    # term_ij = -diff[i,j] / dists[i,j]^2
    # grad[i] = sum_j term_ij
    
    # Vectorized calculation
    # dists_sq = dists_safe ** 2
    # inv_dists_sq = 1.0 / dists_sq
    # Set diagonal to 0 to avoid self-interaction
    np.fill_diagonal(dists_safe, 1.0) # Temporary, we'll mask later
    inv_dists_sq = np.zeros_like(dists_safe)
    mask = dists_safe > 1e-12
    inv_dists_sq[mask] = 1.0 / (dists_safe[mask] ** 2)
    
    # grad = sum_j (-(xi - xj) / r_ij^2)
    # diff is (N, N, 3). We want to multiply diff by inv_dists_sq[..., np.newaxis] and sum over axis 1.
    # term = -diff * inv_dists_sq[..., np.newaxis]
    # grad = np.sum(term, axis=1)
    
    grad = -np.sum(diff * inv_dists_sq[..., np.newaxis], axis=1)
    
    # Project gradient to tangent space: g_t = g - (g . x) x
    dots = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - dots * pts
    
    return energy, grad_tangent

def _fibonacci_points(n, seed_offset=0.0):
    """Generate Fibonacci spiral points on the sphere."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.normal(0, 1, size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _ring_points(n):
    """Generate points in rings, distributing them evenly."""
    pts = []
    # Distribute n points into roughly sqrt(n) rings
    num_rings = int(np.sqrt(n))
    pts_per_ring = n // num_rings
    remainder = n % num_rings
    
    for k in range(num_rings):
        # z coordinate for this ring
        z = 1.0 - 2.0 * (k + 0.5) / num_rings
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        count = pts_per_ring + (1 if k < remainder else 0)
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + k * 0.1 # slight offset
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a), z])
            
    # If we didn't get enough points (due to integer division), add more at poles or random
    while len(pts) < n:
        pts.append([0, 0, 1.0 if len(pts) % 2 == 0 else -1.0])
        
    pts = np.array(pts[:n])
    return pts

def _optimize_single(pts0, max_time=20.0):
    """
    Run L-BFGS-B optimization starting from pts0.
    """
    start_time = time.time()
    
    # Flatten points for scipy
    x0 = pts0.flatten()
    
    def fun_and_grad(x):
        if time.time() - start_time > max_time:
            raise TimeoutError()
        pts = x.reshape(-1, 3)
        # Normalize to ensure we stay on sphere during optimization steps
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        pts = pts / norms
        
        energy, grad = _log_energy_and_grad(pts)
        return energy, grad.flatten()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000})
        best_pts = res.x.reshape(-1, 3)
        best_energy = res.fun
    except TimeoutError:
        # If timeout, return best available? L-BFGS doesn't give intermediate best easily.
        # We'll just return the initial if it fails.
        best_pts = pts0
        best_energy = _log_energy_and_grad(pts0)[0]
    except Exception:
        best_pts = pts0
        best_energy = _log_energy_and_grad(pts0)[0]
        
    # Final polish with projected gradient descent if time remains
    remaining = max_time - (time.time() - start_time)
    if remaining > 1.0:
        best_pts, best_energy = _polish_gradient_descent(best_pts, time_budget=remaining)
        
    return best_pts, best_energy

def _polish_gradient_descent(pts, time_budget=5.0):
    """
    Simple projected gradient descent for final polishing.
    """
    start_time = time.time()
    pts = pts.copy()
    energy, grad = _log_energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    lr = 0.01
    
    while time.time() - start_time < time_budget:
        # Update
        new_pts = pts - lr * grad
        # Normalize
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        new_pts = new_pts / norms
        
        new_energy, new_grad = _log_energy_and_grad(new_pts)
        
        if new_energy < energy:
            pts = new_pts
            energy = new_energy
            grad = new_grad
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
            # Increase LR slightly if improving?
            # lr = min(lr * 1.05, 0.1)
        else:
            # Decrease LR
            lr *= 0.9
            if lr < 1e-6:
                break
                
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # 1. Initial candidate: Fibonacci spiral
    fib_pts = _fibonacci_points(N)
    fib_energy, _ = _log_energy_and_grad(fib_pts)
    best_pts = fib_pts
    best_energy = fib_energy
    
    # 2. Define diverse initializations
    initializations = [
        ("fib", fib_pts),
        ("rand1", _random_points(N, seed=1)),
        ("rand2", _random_points(N, seed=2)),
        ("ring", _ring_points(N)),
    ]
    
    # 3. Optimize each initialization
    # Allocate time based on remaining budget
    num_inits = len(initializations)
    time_per_init = (deadline - start_time - 10.0) / num_inits  # 10s buffer for final checks
    
    for name, init_pts in initializations:
        if time.time() > deadline - 5.0:
            break
            
        # Ensure init is valid
        if not np.all(np.isfinite(init_pts)):
            continue
            
        # Check min dist
        diff = init_pts[:, np.newaxis, :] - init_pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)
        if np.min(dists) < 1e-9:
            continue
            
        try:
            opt_pts, opt_energy = _optimize_single(init_pts, max_time=time_per_init)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts
        except Exception:
            pass

    # 4. Final validation and normalization
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        return _fibonacci_points(N)
    
    # Normalize
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    best_pts = best_pts / norms
    
    # Check min distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        return _fibonacci_points(N)
        
    return best_pts

# EVOLVE-BLOCK-END
