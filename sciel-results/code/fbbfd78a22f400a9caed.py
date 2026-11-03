"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _radius_score(centers):
    """Compute the largest common radius for given centers in [-0.5, 0.5]^2."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check if all centers are inside the square
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # Compute minimum distance to boundary
    boundary_dist = np.min(np.minimum(
        0.5 - centers[:, 0],
        np.minimum(centers[:, 0] + 0.5,
        np.minimum(0.5 - centers[:, 1],
        centers[:, 1] + 0.5))))
    # Compute minimum pairwise distance
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(boundary_dist, min_pair_dist / 2.0)

def _initial_grid():
    """Create a square grid initial guess."""
    m = int(np.ceil(np.sqrt(N)))
    # Use a grid that fits N circles
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _slsqp_optimize(centers, time_budget):
    """Optimize using SLSQP with epigraph formulation."""
    start_time = time.time()
    n = len(centers)
    
    def objective(x):
        # x = [centers (2N), t]
        c = x[:2*n].reshape(n, 2)
        t = x[2*n]
        # Return negative radius (we minimize)
        # Radius is min of all constraints
        # Boundary constraints: 0.5 - |c_i| >= r for each coordinate
        # Pairwise: |c_i - c_j| >= 2r
        
        # Compute all constraint violations
        # Boundary: for each center and each coord, 0.5 - |c_ik| - r >= 0
        # We want to maximize r, so we minimize -r subject to constraints
        
        # Soft approach: penalize violations
        # Actually, let's use a proper formulation
        
        # Radius from boundary
        r_boundary = np.min(np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1])))
        
        # Radius from pairwise
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        
        # The actual radius is min(r_boundary, r_pair)
        # We want to maximize this
        # Use log-sum-exp style soft-min
        # Or just return -min(r_boundary, r_pair) with constraints
        
        # For SLSQP, let's define:
        # Maximize r
        # Subject to:
        #   0.5 - |c_ik| - r >= 0 for all i,k
        #   |c_i - c_j| - 2r >= 0 for all i<j
        
        # But this is non-smooth due to abs and min
        
        # Alternative: minimize -r with constraints
        # Let's try a different approach: minimize -min(r_boundary, r_pair)
        
        return -min(r_boundary, r_pair)
    
    # Constraints for SLSQP
    # This is getting complex. Let me use a simpler approach.
    
    # Actually, let's just use a direct optimization with penalty
    best_center = centers.copy()
    best_score = _radius_score(best_center)
    
    # Use Nelder-Mead or SLSQP with a smooth objective
    def smooth_objective(x):
        c = x.reshape(n, 2)
        # Boundary radius
        r_boundary = np.min(np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1])))
        # Pairwise radius
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        # Soft min using log-sum-exp
        # min(a, b) = -log(exp(-a) + exp(-b)) approximately for large negative
        # Actually: min(a,b) = -logsumexp(-a, -b)
        # But we want to maximize min, so minimize -min
        # -min(a,b) = logsumexp(-a, -b)
        # Let's use a smooth approximation
        beta = 100.0
        a = r_boundary
        b = r_pair
        # -min(a,b) ≈ log(exp(-beta*a) + exp(-beta*b)) / beta ... no
        # min(a,b) = (a+b - |a-b|)/2
        # This is still non-smooth
        
        # Just use the actual min, SLSQP can handle it
        return -min(a, b)
    
    # Bounds: centers must be in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2*n)
    
    try:
        res = minimize(smooth_objective, centers.flatten(), method='SLSQP',
                       bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
        if res.success or res.fun < best_score:
            new_centers = res.x.reshape(n, 2)
            new_score = _radius_score(new_centers)
            if new_score > best_score:
                best_center = new_centers
                best_score = new_score
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    return best_center, best_score, elapsed

def _nelder_mead_optimize(centers, time_budget):
    """Optimize using Nelder-Mead."""
    start_time = time.time()
    n = len(centers)
    
    def objective(x):
        c = x.reshape(n, 2)
        r_boundary = np.min(np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1])))
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        return -min(r_boundary, r_pair)
    
    best_center = centers.copy()
    best_score = _radius_score(best_center)
    
    try:
        res = minimize(objective, centers.flatten(), method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        new_centers = res.x.reshape(n, 2)
        new_score = _radius_score(new_centers)
        if new_score > best_score:
            best_center = new_centers
            best_score = new_score
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    return best_center, best_score, elapsed

def _polish(centers, time_budget):
    """High-precision polish pass."""
    start_time = time.time()
    n = len(centers)
    
    def objective(x):
        c = x.reshape(n, 2)
        r_boundary = np.min(np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1])))
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        return -min(r_boundary, r_pair)
    
    best_center = centers.copy()
    best_score = _radius_score(best_center)
    
    # Multiple restarts with small perturbations
    rng = np.random.RandomState(42)
    for restart in range(10):
        if time.time() - start_time > time_budget - 5:
            break
        # Small perturbation
        perturbed = best_center + rng.normal(0, 0.001, size=(n, 2))
        # Clip to bounds
        perturbed = np.clip(perturbed, -0.49, 0.49)
        
        try:
            res = minimize(objective, perturbed.flatten(), method='SLSQP',
                           bounds=[(-0.5, 0.5)] * (2*n),
                           options={'maxiter': 300, 'ftol': 1e-15})
            new_centers = res.x.reshape(n, 2)
            new_score = _radius_score(new_centers)
            if new_score > best_score:
                best_center = new_centers
                best_score = new_score
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    return best_center, best_score, elapsed

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with grid
    centers = _initial_grid()
    best_centers = centers.copy()
    best_score = _radius_score(best_centers)
    
    # Phase 1: SLSQP optimization
    time_remaining = deadline - time.time()
    if time_remaining > 30:
        centers, score, elapsed = _slsqp_optimize(centers, min(30, time_remaining))
        if score > best_score:
            best_centers = centers
            best_score = score
    
    # Phase 2: Nelder-Mead for refinement
    time_remaining = deadline - time.time()
    if time_remaining > 20:
        centers, score, elapsed = _nelder_mead_optimize(best_centers, min(20, time_remaining))
        if score > best_score:
            best_centers = centers
            best_score = score
    
    # Phase 3: Polish with high precision
    time_remaining = deadline - time.time()
    if time_remaining > 15:
        centers, score, elapsed = _polish(best_centers, min(15, time_remaining))
        if score > best_score:
            best_centers = centers
            best_score = score
    
    return best_centers
# EVOLVE-BLOCK-END
