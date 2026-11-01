"""
Optimized packing of 50 equal circles in a unit square using binary search on radius 
and violation minimization with L-BFGS-B.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
SIDE = 1.0
HALF_SIDE = 0.5

def _compute_radius(centers):
    """
    Compute the maximum radius for a given set of centers.
    r = min(
        min_{i<j} |c_i - c_j| / 2,
        min_i min_k (HALF_SIDE - |c_ik|)
    )
    """
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    min_dist_to_edge = np.min(np.abs(centers)) # This is min distance to 0, but we need distance to boundary
    # Actually, distance to boundary for a point (x,y) in [-0.5, 0.5] is min(0.5-|x|, 0.5-|y|)
    # Let's compute properly
    dists_to_edge = np.min(np.array([
        HALF_SIDE - np.abs(centers[:, 0]),
        HALF_SIDE - np.abs(centers[:, 1])
    ]), axis=1)
    
    if np.any(dists_to_edge < 0):
        # If any center is outside, radius is effectively 0 or invalid, but for scoring we return 0
        return 0.0
        
    min_edge_dist = np.min(dists_to_edge)
    
    # Check pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_edge_dist)

def _cost_function(centers_flat, r):
    """
    Cost function for a fixed target radius r.
    Minimizes the sum of squared violations.
    Violation 1: Pairwise distance < 2r
    Violation 2: Center outside the shrunken square [-0.5+r, 0.5-r]
    """
    centers = centers_flat.reshape(-1, 2)
    
    # Boundary violations
    # Centers must be in [-0.5+r, 0.5-r]
    lower = -0.5 + r
    upper = 0.5 - r
    
    # Calculate how far each center is outside the bounds
    # If x < lower, violation is (lower - x)
    # If x > upper, violation is (x - upper)
    # We use a smooth penalty or just squared distance outside.
    
    viol_x_low = np.maximum(0, lower - centers[:, 0])
    viol_x_high = np.maximum(0, centers[:, 0] - upper)
    viol_y_low = np.maximum(0, lower - centers[:, 1])
    viol_y_high = np.maximum(0, centers[:, 1] - upper)
    
    boundary_cost = np.sum(viol_x_low**2) + np.sum(viol_x_high**2) + \
                    np.sum(viol_y_low**2) + np.sum(viol_y_high**2)
    
    # Pairwise violations
    # Distance between i and j must be >= 2r
    # Violation is max(0, 2r - dist_ij)
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    
    # To be efficient, we can just compute the violations for pairs close to 2r
    # But for N=50, 50x50 is small enough to compute all.
    required_dist = 2 * r
    # dists < required_dist means overlap
    # violation = required_dist - dists (where dists < required_dist)
    overlaps = np.maximum(0, required_dist - dists)
    # We only want to count each pair once, so take upper triangle
    iu = np.triu_indices(N, k=1)
    overlap_vals = overlaps[iu]
    
    pair_cost = np.sum(overlap_vals**2)
    
    return boundary_cost + pair_cost

def _optimize_for_radius(r, initial_centers, time_limit=10.0):
    """
    Try to find a configuration of N centers in the unit square 
    such that all circles of radius r are disjoint and inside.
    Returns the best centers found and the cost.
    """
    start_time = time.time()
    best_centers = initial_centers.copy()
    best_cost = _cost_function(initial_centers.flatten(), r)
    
    # Define bounds for the optimizer
    # Centers must be within [-0.5, 0.5]
    # The optimizer will try to push them into the valid region for radius r
    # But we constrain them to the square [-0.5, 0.5] to keep them valid coordinates
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Use L-BFGS-B
    try:
        res = minimize(
            _cost_function,
            initial_centers.flatten(),
            args=(r,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 1000}
        )
        
        if res.fun < best_cost:
            best_cost = res.fun
            best_centers = res.x.reshape(-1, 2)
            
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > time_limit:
        return best_centers, best_cost, True
        
    # If cost is very small, we might have a feasible solution
    # But we return the centers anyway
    return best_centers, best_cost, False

def _hex_initial():
    """Generate a hexagonal lattice initial guess."""
    # Approximate hexagonal packing
    # Spacing s. For N circles, area ~ N * pi * r^2.
    # Let's just place them in a grid-like hex pattern
    # Side 1. 
    # Let's use a simple grid first, then perturb
    m = int(np.ceil(np.sqrt(N)))
    # Create a grid of m x m points, take first N
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N]
    
    # Perturb to break symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, size=centers.shape)
    return centers

def _random_initial():
    """Generate a random initial guess."""
    rng = np.random.default_rng(123)
    return rng.uniform(-0.45, 0.45, size=(N, 2))

def _grid_initial():
    """Generate a grid initial guess."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def construct():
    """
    Main function to find the best packing.
    """
    start_time = time.time()
    DEADLINE = 140.0 # Safety margin
    
    # Initial configurations
    inits = [
        _grid_initial(),
        _hex_initial(),
        _random_initial()
    ]
    
    # We will binary search on radius.
    # Lower bound: 0
    # Upper bound: 
    # Max possible radius is limited by packing density.
    # For N=50, r ~ 0.07. Let's start high, e.g., 0.1
    r_low = 0.0
    r_high = 0.1
    
    best_centers = inits[0].copy()
    best_r = _compute_radius(best_centers)
    
    # Binary search steps
    # We need to be fast. Each optimization step takes some time.
    # Let's do a coarse binary search first.
    
    # Pre-allocate a list of radii to test if we want to be deterministic
    # But binary search is better.
    
    # To ensure we don't timeout, we limit the number of binary search iterations
    # and the time per iteration.
    
    max_iters = 20 # 20 binary search steps is plenty for precision
    iter_time_limit = 5.0 # 5 seconds per optimization attempt
    
    for _ in range(max_iters):
        if time.time() > DEADLINE - 10: # Keep 10s buffer
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration for r_mid
        # We try multiple initializations to avoid local minima
        success = False
        best_cost_for_r = np.inf
        best_centers_for_r = None
        
        for init in inits:
            if time.time() > DEADLINE - 10:
                break
                
            centers, cost, timed_out = _optimize_for_radius(r_mid, init, time_limit=iter_time_limit)
            
            if cost < best_cost_for_r:
                best_cost_for_r = cost
                best_centers_for_r = centers
                
            # If cost is effectively zero, we found a feasible solution
            # Tolerance for "feasible"
            if cost < 1e-6:
                success = True
                break
        
        if success:
            r_low = r_mid
            # Update best centers and radius
            if best_centers_for_r is not None:
                # Verify the radius of the found centers
                current_r = _compute_radius(best_centers_for_r)
                if current_r > best_r:
                    best_r = current_r
                    best_centers = best_centers_for_r.copy()
        else:
            r_high = r_mid
            
    # Final check: ensure the returned centers are valid and calculate the final score
    # The score is the radius.
    final_r = _compute_radius(best_centers)
    
    # If final_r is 0 or invalid, fall back to a simple valid grid
    if final_r < 1e-5:
        best_centers = _grid_initial()
        final_r = _compute_radius(best_centers)
        
    return best_centers

# EVOLVE-BLOCK-START
# The code above is the implementation.
# To match the "evolve block" structure requested by the prompt's context (though the prompt asks to rewrite from scratch),
# I will wrap the core logic or place it here if strictly required. 
# However, the instruction says "Output the COMPLETE program". 
# The "EVOLVE-BLOCK" markers in the original code suggest that's where changes are expected.
# I will place the main logic inside the EVOLVE-BLOCK markers to be safe, or just output the whole file.
# The prompt says: "Keep the function names and signatures of the seed program."
# Seed program had:
# N = 50
# def construct(): ...
#
# I will output the complete program.
# EVOLVE-BLOCK-END
