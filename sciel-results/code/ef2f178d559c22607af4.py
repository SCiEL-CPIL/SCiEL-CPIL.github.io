import numpy as np
import time

N = 50

def _hex_lattice_init():
    """Initialize with a hexagonal lattice pattern."""
    # Create a grid of points in a hexagonal lattice
    # Spacing 1.0 for initialization, we will scale/adjust later
    # We need enough points to select N=50
    # A circle of radius ~2.5 with spacing 1 will have ~ pi*2.5^2 ~ 20 points?
    # Area of unit circle is pi. Area of hex cell is sqrt(3)/2 * s^2.
    # If s=0.2, area ~ 0.034. 50 * 0.034 ~ 1.7. So radius ~ sqrt(1.7/pi) ~ 0.7.
    # Let's generate a large grid and pick the closest N to origin.
    
    s = 0.1 # Fine spacing for grid generation
    max_r = 2.0
    # Generate hex lattice points
    # x = (i + 0.5*(j%2)) * s
    # y = j * s * sqrt(3)/2
    k = int(max_r / s) + 2
    rows = []
    for j in range(-k, k+1):
        for i in range(-k, k+1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= max_r:
                rows.append((x, y))
    points = np.array(rows)
    
    # Sort by distance to origin
    dists = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(dists, kind='stable')
    points = points[indices[:N]]
    
    # Center the points at origin (they already are, but ensure symmetry if possible)
    # The selection of closest N might be slightly asymmetric.
    # We can try to shift them slightly to center the mass, but origin is usually good.
    return points

def _push_apart(positions, target_r, max_iter=1000):
    """
    Apply push-apart constraints to satisfy:
    1. dist(i, j) >= 2 * target_r
    2. norm(i) <= 1 - target_r
    Returns True if converged (no violations), False otherwise.
    """
    n = len(positions)
    # Precompute indices for pairs
    # Vectorized check is faster
    for _ in range(max_iter):
        # Check boundary
        norms = np.linalg.norm(positions, axis=1)
        max_allowed = 1.0 - target_r
        violations_b = norms > max_allowed
        if np.any(violations_b):
            # Pull in
            # positions[violations_b] *= max_allowed / norms[violations_b]
            # This is a projection
            scale = max_allowed / norms[violations_b]
            positions[violations_b] *= scale[:, np.newaxis]
            
        # Check pairwise distances
        # Compute distance matrix
        # dists[i, j] = norm(pos[i] - pos[j])
        # To save memory/time for N=50, N^2 is 2500, which is small.
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        if min_dist >= 2.0 * target_r:
            return True # Converged
        
        # Identify pairs that are too close
        # We need to push them apart.
        # Iterating over all pairs is O(N^2).
        # For efficiency, we can just do one pass of relaxation.
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = dists[i, j]
                if d < 2.0 * target_r:
                    if d < 1e-10:
                        # Random nudge
                        vec = np.array([1.0, 0.0])
                    else:
                        vec = positions[j] - positions[i]
                        vec /= np.linalg.norm(vec)
                    
                    # Push amount
                    # We want to increase dist by (2r - d)
                    # Split the push between i and j
                    push_amt = (2.0 * target_r - d) / 2.0
                    positions[i] -= vec * push_amt
                    positions[j] += vec * push_amt
                    changed = True
                    
        if not changed:
            # If no pair was identified as violating in the loop (due to floating point or order),
            # but min_dist was low, it might be a numerical issue.
            # However, the loop above updates positions, so dists becomes stale.
            # For N=50, recalculating every iteration is fine.
            # To be safe, if we didn't change anything but min_dist < 2r, break to avoid infinite loop
            # or try a small random perturbation.
            # Actually, the loop above checks dists[i,j] which is from the START of the iteration.
            # It's better to just do a few iterations.
            pass
            
    return False

def _optimize_radius(positions, start_r=None, max_time=5.0):
    """
    Maximize the radius r for the given configuration using the push algorithm.
    """
    n = len(positions)
    if start_r is None:
        # Estimate initial r
        dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        max_r_bound = 1.0 - np.max(np.linalg.norm(positions, axis=1))
        start_r = min(min_d / 2.0, max_r_bound) * 0.9
        
    r = start_r
    # Ensure initial validity
    # If not valid, shrink r
    while not _push_apart(positions.copy(), r, max_iter=50):
        r *= 0.9
        if r < 1e-6:
            return r, positions
    
    best_r = r
    best_pos = positions.copy()
    
    start_time = time.time()
    step = r * 0.01
    min_step = 1e-8
    
    while time.time() - start_time < max_time:
        new_r = r + step
        # Try to push to new_r
        trial_pos = best_pos.copy()
        if _push_apart(trial_pos, new_r, max_iter=2000):
            best_r = new_r
            best_pos = trial_pos
            # Increase step slightly if successful
            step = max(step * 1.05, min_step)
        else:
            # Failed to reach new_r
            # Binary search for the max r in [r, new_r]
            lo, hi = r, new_r
            for _ in range(10):
                mid = (lo + hi) / 2
                trial_pos = best_pos.copy()
                if _push_apart(trial_pos, mid, max_iter=1000):
                    lo = mid
                    best_pos = trial_pos
                else:
                    hi = mid
            best_r = lo
            r = lo
            # Reduce step
            step = max(step * 0.5, min_step)
            
            if step < min_step:
                # Try a small random perturbation to escape local minimum
                noise = np.random.normal(0, 0.01 * best_r, size=(n, 2))
                perturbed = best_pos + noise
                # Check if perturbed is valid for current best_r
                if _push_apart(perturbed, best_r, max_iter=1000):
                    best_pos = perturbed
                    # Try to grow again
                    step = best_r * 0.01
                else:
                    # Perturbation broke validity, revert
                    pass
                    break
                    
    return best_r, best_pos

def construct():
    # Deadline guard
    deadline = time.time() + 110.0
    
    # 1. Initialization
    initial_pos = _hex_lattice_init()
    
    # 2. Initial Optimization
    best_r, best_pos = _optimize_radius(initial_pos, max_time=10.0)
    
    # 3. Iterative Improvement with Perturbation
    # We will try several random perturbations of the best solution
    # and re-optimize.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Perturb the best solution
        # Scale of perturbation should be related to current r
        noise_scale = 0.05 * best_r
        noise = np.random.normal(0, noise_scale, size=(N, 2))
        perturbed_pos = best_pos + noise
        
        # Ensure perturbed positions are roughly inside
        # Project to unit circle if necessary before optimization
        norms = np.linalg.norm(perturbed_pos, axis=1)
        max_allowed = 1.0 - best_r * 0.5 # Loose bound
        too_far = norms > max_allowed
        if np.any(too_far):
            perturbed_pos[too_far] *= (max_allowed / norms[too_far])[:, np.newaxis]
        
        # Optimize from this new starting point
        # Limit time per attempt to ensure we finish
        time_per_attempt = min(10.0, (deadline - time.time()) / 2.0)
        if time_per_attempt < 1.0:
            break
            
        new_r, new_pos = _optimize_radius(perturbed_pos, start_r=best_r * 0.9, max_time=time_per_attempt)
        
        if new_r > best_r:
            best_r = new_r
            best_pos = new_pos
            
    return best_pos

# EVOLVE-BLOCK-END
