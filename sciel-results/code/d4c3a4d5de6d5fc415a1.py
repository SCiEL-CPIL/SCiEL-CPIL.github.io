"""
Optimized logarithmic energy minimization for 150 points on the sphere.
Uses Basin Hopping with L-BFGS-B polishing for high precision.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient for points on the unit sphere.
    pts: (N, 3) array of unit vectors.
    Returns: (energy, grad) where grad is (N, 3) in tangent space.
    """
    n = pts.shape[0]
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist2, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # Note: The problem states sum over all pairs i<j of log(1/d).
    # log(1/d) = -log(d).
    # d_ij = sqrt(dist2_ij).
    # log(d_ij) = 0.5 * log(dist2_ij).
    # So term is -0.5 * log(dist2_ij).
    
    iu = np.triu_indices(n, k=1)
    log_dist2 = np.log(dist2[iu])
    energy = -0.5 * np.sum(log_dist2)
    
    # Gradient: 
    # E = -0.5 * sum_{i!=j} log(||x_i - x_j||^2)
    # dE/dx_i = -0.5 * sum_{j!=i} (1/||x_i-x_j||^2) * 2*(x_i - x_j)
    #          = - sum_{j!=i} (x_i - x_j) / dist2_ij
    #          = sum_{j!=i} (x_j - x_i) / dist2_ij
    
    # Compute sum_{j!=i} (x_j - x_i) / dist2_ij
    # Let A be matrix where A[i,j] = 1/dist2_ij for i!=j, 0 for i=j.
    # Then grad_i = sum_j A[i,j] x_j - x_i * sum_j A[i,j]
    
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad = inv_dist2 @ pts - pts * row_sums
    row_sums = np.sum(inv_dist2, axis=1)
    grad_cart = inv_dist2 @ pts - pts * row_sums[:, np.newaxis]
    
    # Project to tangent space
    # grad_tangent = grad_cart - (grad_cart . x_i) x_i
    dot = np.sum(grad_cart * pts, axis=1, keepdims=True)
    grad = grad_cart - dot * pts
    
    return energy, grad

def _energy_only(pts):
    """Compute just the energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    return -0.5 * np.sum(np.log(dist2[iu]))

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Prevent division by zero
    norms = np.maximum(norms, 1e-12)
    return pts / norms

def _lbfgs_polish(pts, maxiter=200, ftol=1e-15, gtol=1e-12):
    """
    Polish a configuration using L-BFGS-B.
    Uses flat vector representation with projection inside the objective.
    """
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 30})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        # Check if it's valid and better
        e_res = _energy_only(p_res)
        e_orig = _energy_only(pts)
        if e_res < e_orig and np.all(np.isfinite(p_res)):
            return p_res
    except Exception:
        pass
    return pts

def _basin_hopping(pts0, deadline, rng, max_iter=50):
    """
    Basin Hopping algorithm.
    1. Perturb current configuration.
    2. Polish with L-BFGS.
    3. Accept if better (or Metropolis accept).
    """
    current = _project_to_sphere(pts0.copy())
    current_energy = _energy_only(current)
    best = current.copy()
    best_energy = current_energy
    
    T = 0.5  # Initial temperature
    T_min = 1e-6
    cooling = 0.95
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Perturb
        # Move a random subset of points
        n_perturb = rng.randint(5, 15)
        idxs = rng.choice(N, n_perturb, replace=False)
        
        new_pts = current.copy()
        for idx in idxs:
            # Generate random tangent vector
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, new_pts[idx]) * new_pts[idx]
            norm = np.linalg.norm(tangent)
            if norm < 1e-15:
                continue
            tangent /= norm
            new_pts[idx] = new_pts[idx] + T * tangent
        
        new_pts = _project_to_sphere(new_pts)
        
        # Polish
        new_pts = _lbfgs_polish(new_pts, maxiter=100)
        new_energy = _energy_only(new_pts)
        
        # Metropolis Accept
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > T_min and rng.rand() < math.exp(-delta_e / T)):
            current = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()
        
        T *= cooling
        if T < T_min:
            break
            
    return best, best_energy

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)

def _geodesic_dome(n):
    """
    Generate points based on an icosahedron subdivision.
    This provides a highly symmetric initial distribution.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Subdivide faces
    # For simplicity, we'll just use the 12 vertices and then add more by jittering
    # A better approach: Use a library or algorithm to subdivide, but for now,
    # let's use the Fibonacci spiral as primary and just ensure we have diverse starts.
    # Actually, a simple random start is often better for Basin Hopping.
    # We will stick to Fibonacci and Random for initialization, as Geodesic Dome
    # generation is complex to implement correctly for arbitrary N without libraries.
    return _fibonacci_spiral(n)

def construct():
    start_time = time.time()
    total_time_budget = 145.0  # 5s safety margin
    deadline_global = start_time + total_time_budget
    
    rng = np.random.RandomState(777)
    
    # Initial configurations
    starts = []
    starts.append(_fibonacci_spiral(N))
    starts.append(_random_points(N, seed=1))
    starts.append(_random_points(N, seed=2))
    starts.append(_random_points(N, seed=3))
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Local optimization from multiple starts
    # Allocate ~20% of time to this
    phase1_deadline = start_time + 0.2 * total_time_budget
    
    for pts0 in starts:
        if time.time() > phase1_deadline:
            break
        # Quick polish
        pts_opt = _lbfgs_polish(pts0, maxiter=100)
        e = _energy_only(pts_opt)
        if e < best_energy:
            best_energy = e
            best_pts = pts_opt.copy()
            
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_energy = _energy_only(best_pts)
        
    # Phase 2: Basin Hopping
    # Allocate ~60% of time to this
    phase2_deadline = start_time + 0.8 * total_time_budget
    
    # We run basin hopping from the best point found so far
    # We can also run it from other good points if time permits
    bh_pts, bh_energy = _basin_hopping(best_pts, phase2_deadline, rng, max_iter=100)
    
    if bh_energy < best_energy:
        best_energy = bh_energy
        best_pts = bh_pts.copy()
        
    # Phase 3: Final High-Precision Polish
    # Allocate remaining ~20% of time
    if time.time() < deadline_global:
        best_pts = _lbfgs_polish(best_pts, maxiter=500, ftol=1e-18, gtol=1e-15)
        final_energy = _energy_only(best_pts)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = best_pts.copy()
            
    # Safety check
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
        
    best_pts = _project_to_sphere(best_pts)
    return best_pts

# EVOLVE-BLOCK-END
