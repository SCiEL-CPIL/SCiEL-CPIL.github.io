"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    if centres.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check all centres are within [-0.5, 0.5]^3
    if np.any(centres < -0.5 - 1e-12) or np.any(centres > 0.5 + 1e-12):
        return 0.0
    
    # Compute pairwise distances
    n = centres.shape[0]
    # Use broadcasting for efficiency
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Minimum pairwise distance (excluding self)
    np.fill_diagonal(dist, np.inf)
    min_pairwise = np.min(dist)
    
    # Distance to boundary
    boundary_dist = np.min(np.minimum(0.5 - centres, centres + 0.5), axis=1)
    min_boundary = np.min(boundary_dist)
    
    return min(min_pairwise / 2.0, min_boundary)

def _negative_radius(centres):
    """Negative of the radius for minimization."""
    r = _compute_radius(centres)
    return -r

def _objective_smooth(centres, log_k=20.0):
    """Smooth surrogate: minimize -soft_min of all constraints."""
    if centres.shape[0] != N:
        return 1e10
    if not np.all(np.isfinite(centres)):
        return 1e10
    
    n = centres.shape[0]
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances
    boundary_dist = np.min(np.minimum(0.5 - centres, centres + 0.5), axis=1)
    
    # All constraints (half-distances to other spheres, distance to boundary)
    pairwise_half = dist / 2.0
    # Flatten and exclude diagonal
    pairwise_half_flat = pairwise_half[~np.eye(n, dtype=bool)]
    all_constraints = np.concatenate([pairwise_half_flat, boundary_dist])
    
    # Soft minimum via log-sum-exp (negative for min)
    # soft_min(x) = -log(sum(exp(-k*x))) / k
    neg_k_x = -log_k * all_constraints
    # Clip for numerical stability
    neg_k_x = np.clip(neg_k_x, -100, 100)
    log_sum = np.log(np.sum(np.exp(neg_k_x)))
    soft_min = -log_sum / log_k
    
    return -soft_min

def _gradient_smooth(centres, log_k=20.0):
    """Approximate gradient of the smooth objective."""
    eps = 1e-7
    grad = np.zeros_like(centres)
    f0 = _objective_smooth(centres, log_k)
    for i in range(centres.shape[0]):
        for j in range(3):
            old = centres[i, j]
            centres[i, j] = old + eps
            fp = _objective_smooth(centres, log_k)
            centres[i, j] = old - eps
            fm = _objective_smooth(centres, log_k)
            centres[i, j] = old
            grad[i, j] = (fp - fm) / (2 * eps)
    return grad

def _local_optimize(centres, max_iter=500, tol=1e-10):
    """Local optimization using L-BFGS-B with smooth surrogate, then exact polish."""
    # First pass with smooth objective
    result = minimize(
        _objective_smooth,
        centres,
        method='L-BFGS-B',
        jac=_gradient_smooth,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    # Polish with exact objective (nonsmooth but sharper)
    result2 = minimize(
        _negative_radius,
        result.x,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10}
    )
    
    best = result2.x
    best_r = -_negative_radius(best)
    
    if best_r < -_negative_radius(result.x):
        best = result.x
        best_r = -_negative_radius(best)
    
    return best, best_r

def construct():
    """Construct 50 sphere centres in the unit cube to maximize common radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with a good grid-based starting point
    # Use a 3x4x5 = 60 grid, take 50
    m_x, m_y, m_z = 3, 4, 5
    ticks_x = (np.arange(m_x) + 0.5) / m_x - 0.5
    ticks_y = (np.arange(m_y) + 0.5) / m_y - 0.5
    ticks_z = (np.arange(m_z) + 0.5) / m_z - 0.5
    grid = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    centres = grid[:N].copy()
    
    best_centres = centres.copy()
    best_radius = _compute_radius(best_centres)
    
    # Try multiple random perturbations and local optimizations
    np.random.seed(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Create perturbed starting point
        if attempts == 1:
            # Start from the grid
            current = centres.copy()
        else:
            # Random perturbation of best solution
            current = best_centres + np.random.normal(0, 0.02, size=(N, 3))
            # Clip to stay in bounds
            current = np.clip(current, -0.49, 0.49)
        
        # Local optimization
        try:
            optimized, radius = _local_optimize(current, max_iter=200, tol=1e-12)
        except Exception:
            continue
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimized.copy()
        
        # Check if we're close to the known best
        if best_radius > 0.1359:
            break
    
    # Final polish with tighter tolerances if time permits
    if time.time() < deadline - 10.0:
        try:
            final, final_r = _local_optimize(best_centres, max_iter=1000, tol=1e-15)
            if final_r > best_radius:
                best_centres = final
                best_radius = final_r
        except Exception:
            pass
    
    return best_centres
# EVOLVE-BLOCK-END
