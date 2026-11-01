"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

def _compute_radius(centers):
    """Compute the radius for a given set of centers."""
    # Check bounds
    min_bound = np.min(centers, axis=0)
    max_bound = np.max(centers, axis=0)
    # Centers must be in [-0.5, 0.5]
    if np.any(min_bound < -0.5 - 1e-12) or np.any(max_bound > 0.5 + 1e-12):
        return 0.0
    
    # Compute pairwise distances
    n = len(centers)
    if n < 2:
        return 0.5
    
    # Efficient pairwise distance computation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Minimum distance between any pair
    # Exclude self-distances
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Distance to boundary
    # For each center, the distance to the nearest face
    # Face distances: 0.5 - |x|, 0.5 - |y|, 0.5 - |z|
    abs_centers = np.abs(centers)
    boundary_dists = 0.5 - np.max(abs_centers, axis=1)
    min_boundary = np.min(boundary_dists)
    
    # Radius is half the minimum of these
    r = 0.5 * min(min_dist, 2 * min_boundary)
    
    return r

def _soft_min_objective(x, beta=50.0):
    """
    Objective function for optimization.
    x: flattened array of shape (N*3,)
    Returns: negative of the soft-min radius (to be minimized)
    """
    centers = x.reshape(-1, 3)
    
    # Compute pairwise distances
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Exclude self-distances
    np.fill_diagonal(dists, np.inf)
    
    # Soft-min of pairwise distances
    # Using log-sum-exp for numerical stability
    min_dists = np.min(dists, axis=1)
    log_sum = np.log(np.sum(np.exp(-beta * dists), axis=1) / (n - 1))
    soft_min_pair = -np.log(-log_sum) / beta if np.all(np.isfinite(log_sum)) else np.min(dists)
    
    # Actually, let's use a simpler soft-min
    # soft_min = -log(sum(exp(-beta * x))) / beta
    # But we need to be careful with the diagonal
    
    # Recompute properly
    # For each pair (i,j) with i<j, we have distance d_ij
    # We want soft_min of all these distances
    
    # Use a different approach: compute all pairwise distances, 
    # then use soft-min on the upper triangle
    
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    
    if len(pair_dists) == 0:
        soft_min_pair = 1.0
    else:
        # Soft-min: -1/beta * log(sum(exp(-beta * d)))
        # For numerical stability, subtract max
        max_d = np.max(pair_dists)
        exp_terms = np.exp(-beta * (pair_dists - max_d))
        soft_min_pair = max_d - np.log(np.sum(exp_terms)) / beta
    
    # Boundary constraints
    # Distance to boundary for each center
    abs_centers = np.abs(centers)
    # Distance to each face
    dist_x_pos = 0.5 - centers[:, 0]
    dist_x_neg = 0.5 + centers[:, 0]
    dist_y_pos = 0.5 - centers[:, 1]
    dist_y_neg = 0.5 + centers[:, 1]
    dist_z_pos = 0.5 - centers[:, 2]
    dist_z_neg = 0.5 + centers[:, 2]
    
    boundary_dists = np.stack([dist_x_pos, dist_x_neg, dist_y_pos, dist_y_neg, dist_z_pos, dist_z_neg], axis=1)
    min_boundary_dists = np.min(boundary_dists, axis=1)
    
    # Soft-min of boundary distances
    if len(min_boundary_dists) == 0:
        soft_min_boundary = 1.0
    else:
        max_bd = np.max(min_boundary_dists)
        exp_terms_b = np.exp(-beta * (min_boundary_dists - max_bd))
        soft_min_boundary = max_bd - np.log(np.sum(exp_terms_b)) / beta
    
    # The radius is min(soft_min_pair / 2, soft_min_boundary)
    # We want to maximize this, so minimize the negative
    # But we need to combine them properly
    
    # The actual radius is min(pair_min/2, boundary_min)
    # In soft form, we can approximate:
    r_soft = min(soft_min_pair / 2, soft_min_boundary)
    
    # Add penalty for going out of bounds
    penalty = 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        out_of_bounds = np.clip(np.abs(centers) - 0.5, 0, None)
        penalty = np.sum(out_of_bounds**2) * 100
    
    return -r_soft + penalty

def _refine_with_lbfgs(centers, beta=50.0, maxiter=1000):
    """Refine centers using L-BFGS-B."""
    x0 = centers.flatten()
    
    # Define bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    result = minimize(
        _soft_min_objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    refined_centers = result.x.reshape(-1, 3)
    return refined_centers

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with safety margin
    
    # Start with a good initial configuration
    # Use a face-centered cubic (FCC) lattice structure
    
    # Generate FCC lattice points
    # FCC lattice vectors
    a = 0.5  # lattice parameter (will adjust)
    
    # Generate candidate points from FCC lattice
    # FCC has 4 points per unit cell
    # In a cube of side 1, we can fit multiple unit cells
    
    # Let's use a simpler approach: start with a grid and perturb
    # Grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Simulated Annealing phase
    current_temperature = 0.01
    cooling_rate = 0.95
    min_temperature = 0.001
    
    while current_temperature > min_temperature:
        if time.time() > deadline - 30:  # Leave 30 seconds for refinement
            break
        
        # Generate a random perturbation
        idx = np.random.randint(N)
        delta = np.random.normal(0, current_temperature, size=3)
        
        new_centers = best_centers.copy()
        new_centers[idx] += delta
        new_centers[idx] = np.clip(new_centers[idx], -0.49, 0.49)
        
        new_radius = _compute_radius(new_centers)
        
        # Accept if better, or with probability exp((new - old) / temperature)
        if new_radius > best_radius:
            best_centers = new_centers
            best_radius = new_radius
        else:
            accept_prob = np.exp((new_radius - best_radius) / current_temperature)
            if np.random.random() < accept_prob:
                best_centers = new_centers
                best_radius = new_radius
        
        current_temperature *= cooling_rate
    
    # L-BFGS-B refinement phase
    if time.time() < deadline - 10:
        refined = _refine_with_lbfgs(best_centers, beta=50.0, maxiter=500)
        refined_radius = _compute_radius(refined)
        if refined_radius > best_radius:
            best_centers = refined
            best_radius = refined_radius
    
    # Final high-beta refinement
    if time.time() < deadline - 5:
        refined2 = _refine_with_lbfgs(best_centers, beta=100.0, maxiter=300)
        refined_radius2 = _compute_radius(refined2)
        if refined_radius2 > best_radius:
            best_centers = refined2
            best_radius = refined_radius2
    
    # Ensure all centers are strictly inside the cube
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
