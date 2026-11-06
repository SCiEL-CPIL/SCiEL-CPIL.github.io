"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centres):
    """Compute the largest common radius for the given centres."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check inside cube [-0.5, 0.5]^3
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(centres)
    min_boundary = np.min(boundary_dists)
    
    if min_boundary < -1e-12:
        return 0.0

    # Check pairwise distances
    # Vectorized pairwise distances
    # diff[i, j] = c_i - c_j
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_boundary)
    return max(0.0, r)

def _repulsive_energy(x):
    """
    Smooth objective function to maximize spread.
    Minimizes negative sum of inverse squared distances.
    Includes boundary repulsion.
    """
    centres = x.reshape(-1, 3)
    
    # Pairwise repulsion
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-12)
    
    # Sum of inverse squared distances for i < j
    # We can compute the full matrix and take the upper triangle
    # Or just sum all and divide by 2 (diagonal is 0, but we masked it? No, dists_sq diagonal is 0)
    # Let's zero out diagonal to avoid inf
    np.fill_diagonal(dists_sq, 1e12) # Large number so 1/d^2 is ~0
    inv_dists_sq = 1.0 / dists_sq
    pair_energy = np.sum(inv_dists_sq) / 2.0
    
    # Boundary repulsion
    # We want to keep points away from walls.
    # d_wall = 0.5 - |c|
    # If c is close to 0.5, d_wall is small.
    # Use 1/(d_wall + eps)^2 to push away.
    boundary_dists = 0.5 - np.abs(centres)
    # If a point is outside, boundary_dist is negative. We should penalize that heavily.
    # But L-BFGS-B with bounds handles the hard constraint.
    # So we only need to add a soft repulsion for points *inside* but close to wall.
    # Actually, if we use bounds (-0.5, 0.5), the optimizer won't go outside.
    # So we can just use the distance to the wall.
    # To make it smooth and strongly repulsive near the wall:
    boundary_dists = np.maximum(boundary_dists, 1e-12)
    boundary_energy = np.sum(1.0 / (boundary_dists ** 2))
    
    # Total energy to minimize
    # We want to MAXIMIZE spread, so we MINIMIZE negative repulsion?
    # No, repulsion energy is positive and large when close.
    # We want to MINIMIZE the potential energy of repulsion?
    # If spheres repel, the equilibrium is when they are far apart.
    # So we minimize the sum of 1/d^2.
    # As d increases, 1/d^2 decreases. So minimizing this pushes d up.
    
    return pair_energy + 10.0 * boundary_energy

def _optimize_from_start(start_centres, time_limit=10.0):
    """
    Run L-BFGS-B optimization on the repulsive energy.
    """
    start_time = time.time()
    x0 = start_centres.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        return _repulsive_energy(x)
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 2000}
        )
        best_x = result.x
    except Exception:
        best_x = x0
        
    return best_x.reshape(-1, 3)

def _initial_random(seed):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    # Place points randomly, but maybe slightly inset to avoid initial boundary violations
    # though bounds handle it.
    centres = rng.uniform(-0.45, 0.45, size=(N, 3))
    return centres

def _initial_grid():
    """Generate a grid initial configuration."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]

def construct():
    """
    Construct the centres of N equal spheres in the unit cube [-0.5, 0.5]^3
    to maximize the common radius.
    """
    total_start = time.time()
    deadline = total_start + 115.0  # Leave 5s safety margin
    
    best_centres = None
    best_r = -1.0
    
    # Define initializations to try
    # 1. Grid
    # 2. Several random starts
    initializations = []
    initializations.append(_initial_grid())
    
    # Add random starts
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42, 123, 456, 789, 1000, 2024]:
        initializations.append(_initial_random(seed))
        
    # Process each initialization
    for i, init_centres in enumerate(initializations):
        elapsed = time.time() - total_start
        remaining = deadline - elapsed
        
        if remaining < 10.0:
            break
            
        # Allocate time for this optimization
        # We have many starts, so give each a smaller chunk
        # If we have 17 starts, and 115s total, ~6s each.
        # But grid might need more? No, repulsion is fast.
        time_limit = min(remaining / (len(initializations) - i), 15.0)
        
        if time_limit < 2.0:
            # Not enough time for a proper run, just evaluate
            r = _compute_radius(init_centres)
            if r > best_r:
                best_r = r
                best_centres = init_centres.copy()
            continue
            
        optimized_centres = _optimize_from_start(init_centres, time_limit=time_limit)
        
        # Compute exact radius
        r = _compute_radius(optimized_centres)
        
        if r > best_r:
            best_r = r
            best_centres = optimized_centres.copy()
            
    # If we have time left, do a final polish on the best result
    # We can try a second round of optimization with tighter settings
    elapsed = time.time() - total_start
    remaining = deadline - elapsed
    
    if best_centres is not None and remaining > 20.0:
        # Try to polish the best result with a longer run
        polished = _optimize_from_start(best_centres, time_limit=min(remaining - 5.0, 30.0))
        r_polished = _compute_radius(polished)
        if r_polished > best_r:
            best_centres = polished
            best_r = r_polished

    # Final validity check and clipping
    if best_centres is None:
        # Fallback to grid if something went wrong
        best_centres = _initial_grid()
        
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
