"""
Strategy: Basin Hopping with Global Rigid-Body Perturbations.
Instead of Simulated Annealing with small single-point moves, we use Basin Hopping.
The core idea is to repeatedly:
1. Take the current best configuration.
2. Apply a large random rigid-body rotation to ALL points simultaneously.
   This moves the configuration into a completely different region of the 
   configuration space (a different "basin").
3. Run a local optimizer (L-BFGS-B) to find the minimum of the new basin.
4. If the new minimum is lower than the current best, accept it.

This approach is structurally different from SA:
- SA explores locally with small steps and probabilistic acceptance.
- Basin Hopping explores globally via large jumps, relying on the local 
  optimizer to handle the fine-grained search within each basin.

The rigid-body rotation is a valid move because the energy is rotationally 
invariant, but rotating the *entire* set of points relative to the *other* 
points (by applying the rotation to the state vector which is then re-optimized) 
effectively samples different relative geometries. Wait, rotating the entire 
configuration by a fixed rotation doesn't change the energy or the relative 
distances. It's a symmetry.

Correction: To make Basin Hopping work, the perturbation must change the 
relative positions. A common perturbation for point sets is to randomly 
perturb each point independently by a small amount, or to rotate subsets.
However, the "inspiration" mentioned "basin hopping with local L-BFGS-B 
minimization allows escaping shallow local minima... especially with multiple 
diverse initializations".

Let's look at the current program. It uses SA with single-point rotations.
A materially different approach is **Basin Hopping** where the perturbation 
is a **random displacement of each point in its tangent space** (or a random 
rotation of each point individually by a larger angle), followed by L-BFGS-B.

Actually, a very effective and distinct strategy for this type of problem is 
**Multi-start L-BFGS-B with diverse initializations**, combined with a 
**Basin Hopping** loop where the "kick" is a random rotation of individual 
points (but larger than SA steps) or a random perturbation.

Let's implement a clean Basin Hopping algorithm:
1. Initialize with Fibonacci sphere.
2. Loop until time runs out:
   a. Generate a perturbed configuration: For each point, rotate it by a 
      random angle around a random axis. The angle magnitude is controlled 
      by a "kick" parameter.
   b. Run L-BFGS-B to minimize the energy of this perturbed configuration.
   c. If the result is better than the current best, update the best.
   d. Optionally, adapt the kick size (increase if stuck, decrease if improving).

This is structurally different from SA because:
- SA uses a temperature schedule and probabilistic acceptance.
- Basin Hopping uses a deterministic local minimization after each kick, 
  and only accepts if strictly better (greedy basin hopping).
- The search space is explored by jumping between basins rather than 
  random walking within one.

We will use a fixed kick size or a slowly decreasing kick size.
To maximize diversity, we can also periodically restart from a new random 
initialization if the best energy doesn't improve for a long time.

"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 118.0  # Safety margin from 120s

def _log_energy_fast(pts):
    """
    Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|).
    Returns energy only.
    """
    n = len(pts)
    dot = pts @ pts.T
    dist_sq = np.maximum(2.0 - 2.0 * dot, 1e-15)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-18):
        return float('inf')
    terms = -0.5 * np.log(d_sq)
    energy = math.fsum(terms.tolist())
    return energy

def _objective_and_grad(y_flat):
    """
    Objective function for L-BFGS-B.
    Input: y_flat (3N,)
    Output: energy, gradient (3N,)
    """
    n = len(y_flat) // 3
    y = y_flat.reshape(n, 3)
    
    # Normalize to sphere
    norms = np.linalg.norm(y, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = y / norms
    
    # Compute energy
    energy = _log_energy_fast(pts)
    
    if energy == float('inf'):
        return energy, np.zeros_like(y_flat)
        
    # Gradient calculation for the normalized points (tangent space)
    # g_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq_full = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq_full, 1.0)
    weights = 1.0 / dist_sq_full
    np.fill_diagonal(weights, 0.0)
    grad_raw = np.einsum('ij,ijk->ik', weights, diff)
    
    # Project to tangent space
    proj_factor = np.sum(grad_raw * pts, axis=1, keepdims=True)
    grad_tangent = grad_raw - proj_factor * pts
    
    # Map tangent gradient back to raw space gradient
    # dE/dy_i = grad_tangent_i / ||y_i||
    grad_raw_vec = grad_tangent / norms
    
    return energy, grad_raw_vec.flatten()

def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _local_minimize(pts, maxiter=50):
    """
    Run L-BFGS-B to minimize energy starting from pts.
    Returns (pts_opt, energy_opt).
    """
    y_init = pts.flatten()
    try:
        res = minimize(
            _objective_and_grad,
            y_init,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': maxiter,
                'maxfun': maxiter * 2,
                'ftol': 1e-10,
                'gtol': 1e-8,
                'disp': False
            }
        )
        y_opt = res.x.reshape(N, 3)
        pts_opt = _normalize(y_opt)
        e_opt = _log_energy_fast(pts_opt)
        if not np.isfinite(e_opt):
            return pts, _log_energy_fast(pts)
        return pts_opt, e_opt
    except Exception:
        return pts, _log_energy_fast(pts)

def construct():
    start_time = time.time()
    
    # Initial candidate: Fibonacci sphere
    best_pts = _fibonacci_sphere(N)
    best_energy = _log_energy_fast(best_pts)
    
    # First, do a local minimization on the initial guess
    best_pts, best_energy = _local_minimize(best_pts, maxiter=100)
    
    # Basin Hopping parameters
    # kick_size controls the magnitude of the random rotation for each point
    # We start with a moderate kick and adapt it
    kick_size = 0.2  # Radians
    rng = np.random.default_rng(42)
    
    # Stagnation tracking
    last_improvement_time = start_time
    stagnation_counter = 0
    max_stagnation_time = 10.0  # Seconds
    
    while time.time() < DEADLINE:
        elapsed = time.time() - start_time
        
        # Check for stagnation: if no improvement for max_stagnation_time,
        # try a larger kick or a new random start
        if time.time() - last_improvement_time > max_stagnation_time:
            # Increase kick size to jump further
            kick_size *= 1.5
            if kick_size > 1.0:
                # If kick is too large, restart from a new random configuration
                new_init = _random_uniform_sphere(N, seed=int(time.time()*1000) % 10000)
                best_pts, best_energy = _local_minimize(new_init, maxiter=50)
                kick_size = 0.2
                last_improvement_time = time.time()
                continue
        
        # Generate perturbation: rotate each point by a random angle around a random axis
        perturbed_pts = best_pts.copy()
        
        # Vectorized rotation generation
        # Random axes
        axes = rng.normal(size=(N, 3))
        axis_norms = np.linalg.norm(axes, axis=1, keepdims=True)
        axis_norms = np.maximum(axis_norms, 1e-10)
        axes = axes / axis_norms
        
        # Random angles
        angles = rng.uniform(-kick_size, kick_size, N)
        
        # Apply rotation to each point
        # Using Rodrigues' rotation formula
        # v_rot = v * cos(a) + (k x v) * sin(a) + k * (k . v) * (1 - cos(a))
        c = np.cos(angles)[:, np.newaxis]
        s = np.sin(angles)[:, np.newaxis]
        k = axes
        
        # k x v
        cross_k_v = np.cross(k, perturbed_pts)
        # k . v
        dot_k_v = np.sum(k * perturbed_pts, axis=1, keepdims=True)
        
        perturbed_pts = perturbed_pts * c + cross_k_v * s + k * dot_k_v * (1 - c)
        
        # Normalize
        perturbed_pts = _normalize(perturbed_pts)
        
        # Local minimization
        candidate_pts, candidate_energy = _local_minimize(perturbed_pts, maxiter=30)
        
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts
            last_improvement_time = time.time()
            # Decrease kick size slightly as we improve
            kick_size *= 0.9
            stagnation_counter = 0
        else:
            stagnation_counter += 1
    
    # Final polish with more iterations
    if time.time() < DEADLINE - 5:
        best_pts, best_energy = _local_minimize(best_pts, maxiter=200)
        
    # Final normalization and check
    best_pts = _normalize(best_pts)
    
    return best_pts
