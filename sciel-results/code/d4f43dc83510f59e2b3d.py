"""
Strategy: Simulated Annealing (SA) for global exploration on the sphere,
combined with Riemannian Conjugate Gradient (RCG) for local exploitation.

The current program relies on deterministic gradient-based methods (RCG) from
multiple initializations. This approach is susceptible to getting stuck in
local minima because the logarithmic energy landscape for N=150 is likely
rugged with many local minima. 

This new strategy uses Simulated Annealing to allow "uphill" moves (increasing
energy) with a probability that decreases over time. This allows the search
to escape local minima. To ensure fast convergence to the bottom of each
basin of attraction, we apply a short RCG polish after every accepted SA move.

Key components:
1. _log_energy_and_grad_sphere: Efficient computation of energy and gradient.
2. _exp_map: Exponential map for moving points on the sphere.
3. _riemannian_cg: Local optimizer to polish configurations.
4. _simulated_annealing: Global optimizer using random perturbations (rotations/swaps)
   and Metropolis acceptance criterion.
5. construct: Orchestrates the SA process with a time limit, returning the best
   configuration found.
"""

import numpy as np
import time
import math
import random

N = 150

# EVOLVE-BLOCK-START

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its Euclidean gradient for points on the sphere.
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3)
    """
    # Compute pairwise differences
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal to avoid log(0)
    mask = ~np.eye(N, dtype=bool)
    
    # Safe distance for computation
    safe_dist = np.where(mask, dist, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient: 
    # d/dx_i sum_{j!=i} -log(||x_i - x_j||)
    # = sum_{j!=i} - (1/||x_i - x_j||) * (x_i - x_j) / ||x_i - x_j||
    # = sum_{j!=i} -(x_i - x_j) / ||x_i - x_j||^2
    
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    # weighted_diff[i, j] = (x_i - x_j) / ||x_i - x_j||^2
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    
    # Sum over j for each i. Note that the term for j=i is 0 because mask is False there (inv_dist_sq is 0).
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _riemannian_grad(x, grad_euclidean):
    """
    Project Euclidean gradient onto the tangent space of the sphere at x.
    T_x S^2 = {v : <v, x> = 0}
    """
    proj = np.sum(grad_euclidean * x, axis=1, keepdims=True)
    return grad_euclidean - proj * x

def _exp_map(x, v):
    """
    Exponential map on the unit sphere.
    x: (N, 3) points on sphere
    v: (N, 3) tangent vectors
    Returns: new points on sphere
    """
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    norm_v_safe = np.where(norm_v < 1e-12, 1e-12, norm_v)
    sin_norm = np.sin(norm_v_safe)
    cos_norm = np.cos(norm_v_safe)
    
    small = norm_v < 1e-12
    sin_norm[small] = 0.0
    cos_norm[small] = 1.0
    
    factor = sin_norm / norm_v_safe
    factor[small] = 1.0
    
    new_x = cos_norm * x + factor * v
    new_x /= np.linalg.norm(new_x, axis=1, keepdims=True)
    return new_x

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def _random_sphere(n, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _riemannian_cg(x, max_iter=50, tol=1e-8, time_limit=2.0):
    """
    Riemannian Conjugate Gradient method on the sphere.
    Used for local polishing.
    """
    start_time = time.time()
    energy, grad_eucl = _log_energy_and_grad_sphere(x)
    grad_tan = _riemannian_grad(x, grad_eucl)
    
    if np.linalg.norm(grad_tan) < tol:
        return x, energy
    
    p = -grad_tan
    p_norm = np.linalg.norm(p)
    if p_norm > 0:
        p /= p_norm
    else:
        return x, energy
        
    best_x = x.copy()
    best_energy = energy
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Line search: backtracking
        t = 1.0
        c1 = 1e-4
        alpha = 1e-3
        grad_dot_p = np.sum(grad_tan * p)
        
        for _ in range(20):
            x_new = _exp_map(x, t * p)
            e_new, grad_eucl_new = _log_energy_and_grad_sphere(x_new)
            
            if e_new < energy + c1 * t * grad_dot_p:
                break
            t *= alpha
            
        x = x_new
        energy = e_new
        grad_eucl = grad_eucl_new
        grad_tan_new = _riemannian_grad(x, grad_eucl)
        
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
            
        if np.linalg.norm(grad_tan_new) < tol:
            break
            
        # Polak-Ribiere beta
        grad_diff = np.sum(grad_tan_new * (grad_tan_new - grad_tan))
        grad_sq = np.sum(grad_tan * grad_tan)
        beta = grad_diff / (grad_sq + 1e-12)
        beta = max(0.0, beta)
        
        p_new = -grad_tan_new + beta * p
        p_new = _riemannian_grad(x, p_new)
        
        p_new_norm = np.linalg.norm(p_new)
        if p_new_norm < 1e-12:
            p_new = -grad_tan_new
            p_new = _riemannian_grad(x, p_new)
            p_new_norm = np.linalg.norm(p_new)
            
        if p_new_norm > 0:
            p = p_new / p_new_norm
        else:
            break
            
        grad_tan = grad_tan_new
        
    return best_x, best_energy

def _random_rotation():
    """Generate a random rotation matrix (3x3)."""
    # Use QR decomposition of a random Gaussian matrix
    A = np.random.randn(3, 3)
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det=1)
    d = np.sign(np.linalg.det(Q))
    Q[:, -1] *= d
    return Q

def _simulated_annealing(x_init, time_limit=120.0, initial_temp=10.0, cooling_rate=0.999):
    """
    Simulated Annealing for minimizing logarithmic energy.
    """
    start_time = time.time()
    x = x_init.copy()
    energy, _ = _log_energy_and_grad_sphere(x)
    
    # Polish initial state
    x, energy = _riemannian_cg(x, max_iter=50, time_limit=2.0)
    
    best_x = x.copy()
    best_energy = energy
    current_temp = initial_temp
    
    # Pre-allocate random numbers for speed? No, np.random is fast enough.
    
    while time.time() - start_time < time_limit:
        # Generate a random perturbation
        # Option 1: Rotate a random subset of points
        # Option 2: Swap two random points
        # Option 3: Small random step for a random point
        
        # Let's use a mix. 
        # 50% chance: Small step for 1 point
        # 30% chance: Rotation of 2 points
        # 20% chance: Swap 2 points (rarely helps with log energy as it's symmetric, 
        # but swapping positions of points i and j is same as identity if labels don't matter. 
        # Wait, the energy is sum over pairs. The set of points is what matters. 
        # Swapping labels doesn't change the set. So swapping is useless.
        # Instead, let's move two points towards each other or away? 
        # Better: Randomly perturb 2 points.
        
        r = np.random.rand()
        x_new = x.copy()
        
        if r < 0.5:
            # Perturb 1 point
            idx = np.random.randint(N)
            # Random tangent vector
            v = np.random.randn(3)
            v -= np.dot(v, x[idx]) * x[idx]
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-12:
                v /= v_norm
                # Step size proportional to temperature? Or fixed?
                # Let's use a fixed small step for stability, SA handles acceptance.
                step_size = 0.1
                x_new[idx] = _exp_map(x[idx], step_size * v)[0]
        elif r < 0.8:
            # Perturb 2 points
            idx1, idx2 = np.random.choice(N, 2, replace=False)
            v1 = np.random.randn(3)
            v1 -= np.dot(v1, x[idx1]) * x[idx1]
            v1 /= np.linalg.norm(v1)
            
            v2 = np.random.randn(3)
            v2 -= np.dot(v2, x[idx2]) * x[idx2]
            v2 /= np.linalg.norm(v2)
            
            step_size = 0.1
            x_new[idx1] = _exp_map(x[idx1], step_size * v1)[0]
            x_new[idx2] = _exp_map(x[idx2], step_size * v2)[0]
        else:
            # Larger perturbation: Rotate a random pair by a larger angle?
            # Or just a larger step for one point.
            idx = np.random.randint(N)
            v = np.random.randn(3)
            v -= np.dot(v, x[idx]) * x[idx]
            v /= np.linalg.norm(v)
            step_size = 0.5
            x_new[idx] = _exp_map(x[idx], step_size * v)[0]
        
        # Evaluate new energy
        e_new, _ = _log_energy_and_grad_sphere(x_new)
        delta_e = e_new - energy
        
        # Metropolis criterion
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / current_temp):
            x = x_new
            energy = e_new
            
            # Polish the new state
            # Use a very short time limit for polish to keep SA moving
            x_polished, e_polished = _riemannian_cg(x, max_iter=10, time_limit=0.5)
            
            if e_polished < energy:
                x = x_polished
                energy = e_polished
            
            if energy < best_energy:
                best_energy = energy
                best_x = x.copy()
        
        # Cool down
        current_temp *= cooling_rate
        if current_temp < 1e-6:
            # Reheat if stuck? Or just stop cooling?
            # For now, keep it low.
            pass
        
    return best_x, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = float('inf')
    best_x = None
    
    # Initial configurations
    # Use diverse starts
    inits = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=42),
        _random_sphere(N, seed=123),
    ]
    
    # Run SA for each initial configuration
    # Allocate time budget to each
    # We have ~145s. Let's give each run ~40s.
    # But we should stop early if we find a very good solution.
    
    time_per_run = 40.0
    
    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        x_res, e_res = _simulated_annealing(x_init, time_limit=time_per_run)
        
        if e_res < best_energy:
            best_energy = e_res
            best_x = x_res
            
    # Final polish with more time
    if best_x is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5:
            x_final, e_final = _riemannian_cg(best_x, max_iter=200, time_limit=remaining_time)
            if e_final < best_energy:
                best_energy = e_final
                best_x = x_final
    
    if best_x is None:
        best_x = inits[0]
        
    return best_x

# EVOLVE-BLOCK-END
