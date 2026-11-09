import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    """Check if the packing is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
            
    # Check no overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_dx = (sides[i] + sides[j])/2 - tol
            min_dy = (sides[i] + sides[j])/2 - tol
            if dx < min_dx and dy < min_dy:
                return False
    return True

def _objective(centers, sides):
    """Return sum of sides if valid, else a large penalty."""
    n = len(sides)
    tol = 1e-7
    
    # Check bounds
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return -1e6
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return -1e6
        if s <= 0:
            return -1e6
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_dx = (sides[i] + sides[j])/2 - tol
            min_dy = (sides[i] + sides[j])/2 - tol
            if dx < min_dx and dy < min_dy:
                return -1e6
    
    return np.sum(sides)

def _smooth_objective_flat(x):
    """
    Smooth objective function for Nelder-Mead.
    x is a flat array of length 30: [cx0, cy0, s0, cx1, cy1, s1, ...]
    Returns negative sum of sides + penalties (to be minimized).
    """
    n = 10
    centers = x[0:20].reshape(n, 2)
    sides = x[20:30]
    
    # Ensure sides are positive
    sides = np.maximum(sides, 1e-6)
    
    # Reconstruct x with clipped sides for consistency in penalty? 
    # Actually, just use the raw sides for the objective value but penalize if <= 0
    # Nelder-Mead handles variables, so we should just penalize invalid states smoothly.
    
    penalty = 0.0
    total_side_sum = 0.0
    
    # Boundary penalties
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        total_side_sum += s
        
        # Smooth boundary violation
        # We want cx - s/2 >= 0 and cx + s/2 <= 1
        # Violation 1: cx - s/2 < 0  => max(0, s/2 - cx)
        # Violation 2: cx + s/2 > 1  => max(0, cx + s/2 - 1)
        # Use a smooth max: log(1+exp(k*val))/k approximates max(0, val)
        # Or just use squared penalties for simplicity and differentiability
        v1 = max(0, s/2 - cx)
        v2 = max(0, cx + s/2 - 1)
        v3 = max(0, s/2 - cy)
        v4 = max(0, cy + s/2 - 1)
        
        penalty += (v1**2 + v2**2 + v3**2 + v4**2) * 100.0
        
        # Penalize negative sides
        if s <= 0:
            penalty += (s**2) * 100.0

    # Overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            
            # Required separation
            req_dx = (sides[i] + sides[j])/2
            req_dy = (sides[i] + sides[j])/2
            
            # Overlap amount
            # If dx < req_dx AND dy < req_dy, there is overlap.
            # Overlap area is not directly penalized, but we can penalize the "penetration" depth.
            # Let overlap_x = max(0, req_dx - dx)
            # Let overlap_y = max(0, req_dy - dy)
            # If both are > 0, they overlap.
            # A simple penalty is (overlap_x * overlap_y) or sum of squares.
            # To make it smooth and push them apart, we can penalize the square of the overlap dimensions.
            
            ox = max(0, req_dx - dx)
            oy = max(0, req_dy - dy)
            
            # If they overlap in both axes, penalize the area of intersection of the "required separation boxes"
            # This creates a gradient that pushes them apart until ox or oy becomes 0.
            penalty += (ox**2 + oy**2) * 10.0
            
    # We want to maximize sum of sides, so minimize -sum(sides) + penalty
    return -total_side_sum + penalty

def _local_search_smooth(centers, sides, time_limit=50.0):
    """
    Nelder-Mead local search on the smooth objective.
    """
    start_time = time.time()
    n = 10
    
    # Flatten variables
    x0 = np.concatenate([centers.flatten(), sides])
    
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _objective(best_centers, best_sides)
    
    if best_score < 0:
        return best_centers, best_sides, best_score
        
    # Nelder-Mead parameters
    options = {
        'maxiter': 5000,
        'maxfev': 10000,
        'xatol': 1e-8,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    try:
        # Use a timeout wrapper? Nelder-Mead doesn't have a built-in time limit.
        # We can break out if it takes too long, but minimize runs to completion.
        # Given the small dimension (30), 5000 iterations should be fast (<1s).
        # If it's too slow, we reduce maxiter.
        result = minimize(
            _smooth_objective_flat, 
            x0, 
            method='Nelder-Mead', 
            options=options
        )
        
        x_res = result.x
        new_centers = x_res[0:20].reshape(n, 2)
        new_sides = x_res[20:30]
        
        # Clamp sides to be positive
        new_sides = np.maximum(new_sides, 1e-6)
        
        # Clamp centers to be within [0,1] roughly, to help validity check
        for i in range(n):
            s = new_sides[i]
            new_centers[i, 0] = np.clip(new_centers[i, 0], 0, 1)
            new_centers[i, 1] = np.clip(new_centers[i, 1], 0, 1)
            
        new_score = _objective(new_centers, new_sides)
        
        if new_score > best_score:
            best_centers = new_centers
            best_sides = new_sides
            best_score = new_score
            
    except Exception as e:
        pass
        
    return best_centers, best_sides, best_score

def _generate_initial_config():
    """Generate the known 3.0 packing."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # Bottom half: 2 squares of side 0.5
    sides[0] = 0.5
    sides[1] = 0.5
    centers[0, 0] = 0.25
    centers[0, 1] = 0.25
    centers[1, 0] = 0.75
    centers[1, 1] = 0.25
    
    # Top half: 8 squares of side 0.25
    sides[2] = 0.25
    centers[2, 0] = 0.125
    centers[2, 1] = 0.625
    
    sides[3] = 0.25
    centers[3, 0] = 0.375
    centers[3, 1] = 0.625
    
    sides[4] = 0.25
    centers[4, 0] = 0.125
    centers[4, 1] = 0.875
    
    sides[5] = 0.25
    centers[5, 0] = 0.375
    centers[5, 1] = 0.875
    
    sides[6] = 0.25
    centers[6, 0] = 0.625
    centers[6, 1] = 0.625
    
    sides[7] = 0.25
    centers[7, 0] = 0.875
    centers[7, 1] = 0.625
    
    sides[8] = 0.25
    centers[8, 0] = 0.625
    centers[8, 1] = 0.875
    
    sides[9] = 0.25
    centers[9, 0] = 0.875
    centers[9, 1] = 0.875
    
    return centers, sides

def squares_in_square10() -> tuple:
    """
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    
    Uses Nelder-Mead local search with a smooth objective to try to improve upon the known 3.0 packing.
    Falls back to the known packing if no improvement is found.
    """
    deadline = time.time() + 580  # Safety margin before 630s limit
    
    # Start with known 3.0 packing
    centers, sides = _generate_initial_config()
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _objective(best_centers, best_sides)
    
    # Try local search with different random seeds and perturbations
    # We use a few different starting points to escape the 3.0 local minimum
    seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(seed)
        
        # Perturb the initial configuration
        # Strategy 1: Small perturbation
        # Strategy 2: Larger perturbation to break symmetry
        # Strategy 3: Randomize sizes slightly
        
        if seed % 2 == 0:
            # Small perturbation
            perturbed_centers = centers + rng.normal(0, 0.01, (10, 2))
            perturbed_sides = sides * (1 + rng.normal(0, 0.02, 10))
        else:
            # Larger perturbation
            perturbed_centers = centers + rng.normal(0, 0.05, (10, 2))
            perturbed_sides = sides * (1 + rng.normal(0, 0.1, 10))
            
        perturbed_sides = np.clip(perturbed_sides, 0.01, 1.0)
        
        # Clamp centers to valid range
        for i in range(10):
            s = perturbed_sides[i]
            perturbed_centers[i, 0] = np.clip(perturbed_centers[i, 0], s/2, 1 - s/2)
            perturbed_centers[i, 1] = np.clip(perturbed_centers[i, 1], s/2, 1 - s/2)
        
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            break
        
        search_time = min(50.0, remaining_time - 5)
        if search_time < 5:
            break
        
        new_centers, new_sides, new_score = _local_search_smooth(
            perturbed_centers, perturbed_sides, time_limit=search_time
        )
        
        if new_score > best_score:
            best_centers = new_centers
            best_sides = new_sides
            best_score = new_score
            
    # Final validation
    if not _is_valid(best_centers, best_sides):
        # Fall back to known valid packing
        best_centers, best_sides = _generate_initial_config()
    
    return best_centers, best_sides
