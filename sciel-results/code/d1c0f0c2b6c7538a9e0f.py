# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _normalize(v):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms

def _vectorized_circumradii(points, i, neighbor_indices):
    """
    Compute circumradii for all pairs of neighbors for point i.
    points: (N, 3)
    i: int
    neighbor_indices: array of indices
    Returns: array of circumradii
    """
    p_i = points[i]
    p_neighbors = points[neighbor_indices]
    n_neigh = len(p_neighbors)
    
    if n_neigh < 2:
        return np.array([])
    
    # Create meshgrid of indices
    # We want pairs (j, k) with j < k
    idx_j, idx_k = np.triu_indices(n_neigh, k=1)
    
    p_j = p_neighbors[idx_j] # (M, 3)
    p_k = p_neighbors[idx_k] # (M, 3)
    
    # v1 = p_i - p_j
    # v2 = p_i - p_k
    v1 = p_i[None, :] - p_j # (M, 3)
    v2 = p_i[None, :] - p_k # (M, 3)
    
    c = np.cross(v1, v2) # (M, 3)
    c_norms = np.linalg.norm(c, axis=1, keepdims=True)
    c_norms = np.where(c_norms < 1e-12, 1.0, c_norms)
    c = c / c_norms
    
    cos_r = np.dot(c, p_i[None, :].T).flatten()
    cos_r = np.clip(cos_r, -1.0, 1.0)
    
    # Handle degeneracies where cross product is zero
    # If norm was small, the points are collinear. 
    # In that case, the circumradius is effectively infinite or the max distance.
    # For the purpose of the objective, we can assign a large value or the max pairwise distance.
    # Let's check which ones were degenerate
    degenerate = c_norms.flatten() < 1e-8
    if np.any(degenerate):
        # For degenerate cases, use max distance to the two neighbors
        # d1 = arccos(p_i . p_j), d2 = arccos(p_i . p_k)
        # R ~ max(d1, d2) / 2 is a lower bound, but actually if they are collinear on the great circle,
        # the circumcenter is undefined.
        # A safe fallback for the objective is to use a large penalty or the max distance.
        # Let's use the max angular distance between the three points.
        d_ij = np.arccos(np.clip(np.dot(p_i, p_j[degenerate]), -1, 1))
        d_ik = np.arccos(np.clip(np.dot(p_i, p_k[degenerate]), -1, 1))
        d_jk = np.arccos(np.clip(np.dot(p_j[degenerate], p_k[degenerate]), -1, 1))
        max_dists = np.max(np.vstack([d_ij, d_ik, d_jk]), axis=0)
        # The circumradius of a degenerate triangle is not well-defined, 
        # but for covering, the "empty cap" radius would be large.
        # We assign the max distance to push the optimizer to fix the degeneracy.
        radii = np.arccos(cos_r)
        radii[degenerate] = max_dists
        return radii
    
    return np.arccos(cos_r)

def _objective(x):
    """
    Objective: sum of squared Voronoi inradii (approximated by soft-max of circumradii).
    """
    points = _normalize(x.reshape(N, 3))
    
    # Compute pairwise distances to find neighbors
    cos_dists = np.clip(points @ points.T, -1.0, 1.0)
    dists = np.arccos(cos_dists)
    np.fill_diagonal(dists, 0.0)
    
    total_obj = 0.0
    k_neighbors = 12 # Sufficient for N=50
    
    for i in range(N):
        # Get k nearest neighbors
        idx = np.argsort(dists[i])[:k_neighbors]
        
        radii = _vectorized_circumradii(points, i, idx)
        
        if len(radii) == 0:
            continue
            
        # Soft max
        temp = 0.05
        # Avoid overflow in exp
        max_r = np.max(radii)
        soft_max = max_r + temp * np.log(np.sum(np.exp((radii - max_r) / temp)))
        
        total_obj += soft_max ** 2
        
    return total_obj

def _gradient_finite_diff(x, eps=1e-6):
    """
    Finite difference gradient.
    """
    grad = np.zeros_like(x)
    f0 = _objective(x)
    
    # Use central differences for better accuracy
    for j in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        
        f_plus = _objective(x_plus)
        f_minus = _objective(x_minus)
        
        grad[j] = (f_plus - f_minus) / (2.0 * eps)
        
    return grad

def construct():
    start_time = time.time()
    max_time = 115.0
    
    # Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    x0 = points.flatten().copy()
    
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    # Phase 1: Coarse optimization
    # Use L-BFGS-B with finite difference gradient
    # We provide the gradient to ensure consistent steps
    try:
        result = minimize(
            _objective,
            x0,
            jac=_gradient_finite_diff,
            method='L-BFGS-B',
            options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        if result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
        
    if time.time() - start_time > max_time - 40:
        return _normalize(best_x.reshape(N, 3))
        
    # Phase 2: Local search with restarts
    rng = np.random.RandomState(42)
    for restart in range(15):
        # Create perturbation
        scale = 0.1 * (1.0 - restart / 20.0)
        x_perturbed = best_x + rng.randn(N * 3) * scale
        
        try:
            result = minimize(
                _objective,
                x_perturbed,
                jac=_gradient_finite_diff,
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 1000, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
            
        if time.time() - start_time > max_time - 20:
            break
            
    # Phase 3: Final fine-tuning
    if time.time() - start_time < max_time - 5:
        try:
            result = minimize(
                _objective,
                best_x,
                jac=_gradient_finite_diff,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
            
    points = _normalize(best_x.reshape(N, 3))
    
    # Validation
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-6):
        # Fallback
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return points

# EVOLVE-BLOCK-END
