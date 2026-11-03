"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import time


def _hex_initial(n, r_target):
    """Generate hexagonal lattice points inside unit disk, sorted by radius."""
    # Estimate spacing from target radius
    spacing = 2.0 * r_target
    if spacing <= 0:
        spacing = 2.0 / (np.sqrt(n) + 1.0)
    
    k = int(1.0 / (spacing * np.sqrt(3.0) / 2.0)) + 2
    rows = []
    for j in range(-k, k + 1):
        y = j * spacing * np.sqrt(3.0) / 2.0
        offset = 0.5 * spacing if (j % 2) else 0.0
        for i in range(-k, k + 1):
            x = (i + 0.5) * spacing + offset
            rows.append([x, y])
    
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - r_target
    points = points[mask]
    
    if len(points) < n:
        # Fallback: denser grid
        xs = np.linspace(-0.95, 0.95, 30)
        ys = np.linspace(-0.95, 0.95, 30)
        grid = np.array([[x, y] for x in xs for y in ys])
        radius = np.hypot(grid[:, 0], grid[:, 1])
        mask = radius <= 1.0 - r_target
        grid = grid[mask]
        if len(grid) >= n:
            points = grid
        else:
            # Very dense grid
            xs = np.linspace(-0.95, 0.95, 50)
            ys = np.linspace(-0.95, 0.95, 50)
            grid = np.array([[x, y] for x in xs for y in ys])
            radius = np.hypot(grid[:, 0], grid[:, 1])
            mask = radius <= 1.0 - r_target
            grid = grid[mask]
            points = grid
    
    if len(points) < n:
        raise ValueError("Cannot generate enough initial points")
    
    # Sort by radius to get well-distributed points
    radius = np.hypot(points[:, 0], points[:, 1])
    order = np.argsort(radius, kind='stable')
    points = points[order[:n]]
    
    return points


def _compute_radius(centres):
    """Compute the valid packing radius for given centres."""
    n = len(centres)
    if n < 2:
        return 1.0 - np.max(np.linalg.norm(centres, axis=1))
    
    dists = pdist(centres)
    min_pair_dist = np.min(dists)
    boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    if not np.isfinite(r) or r <= 0:
        return 0.0
    
    return r


def _residuals(x, n, target_r):
    """
    Residuals for least_squares optimization.
    
    We want to maximize the minimum of:
    - Pairwise distances / 2
    - Distance to boundary (1 - |c_i|)
    
    Strategy: Minimize violations of the constraint that all distances
    should be at least 2*target_r (pairs) and target_r (boundary).
    
    For pairs with d_ij < 2*target_r: residual = 2*target_r - d_ij
    For boundary with 1-|c_i| < target_r: residual = target_r - (1-|c_i|)
    
    Additionally, we add a small regularization to encourage spreading out.
    """
    centres = x.reshape(n, 2)
    
    # Pairwise distances
    dists = pdist(centres)
    
    # Boundary distances
    norms = np.linalg.norm(centres, axis=1)
    boundary_dists = 1.0 - norms
    
    # Pair residuals: violations of minimum distance constraint
    pair_residuals = np.maximum(0.0, 2.0 * target_r - dists)
    
    # Boundary residuals: violations of containment constraint
    boundary_residuals = np.maximum(0.0, target_r - boundary_dists)
    
    # Small regularization: encourage points to be well-distributed
    # Add a tiny penalty for points too close to center (to avoid clustering)
    center_penalty = 0.01 * norms  # Encourage spreading from center
    
    return np.concatenate([pair_residuals, boundary_residuals, center_penalty])


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses iterative target radius increase with scipy.optimize.least_squares.
    """
    n = N
    start_time = time.time()
    max_time = 100  # Leave margin for evaluation overhead
    
    # Start with a reasonable initial radius estimate
    # Hexagonal packing density is ~0.9069
    # Area of unit circle = pi, area of n circles = n * pi * r^2
    # r^2 <= 0.9069 * pi / (n * pi) = 0.9069 / n
    r_est = np.sqrt(0.9069 / n) * 0.95  # 5% margin
    r_est = min(r_est, 0.055)  # Cap at known best + small margin
    
    best_centres = None
    best_r = 0.0
    current_r = r_est
    
    # Generate initial configuration
    try:
        x0 = _hex_initial(n, current_r)
    except Exception:
        # Fallback to simple grid
        xs = np.linspace(-0.9, 0.9, 20)
        ys = np.linspace(-0.9, 0.9, 20)
        x0 = np.array([[x, y] for x in xs for y in ys])[:n]
    
    # Iterative optimization with increasing target radius
    max_iterations = 15
    target_r = current_r * 1.05
    
    for iteration in range(max_iterations):
        if time.time() - start_time > max_time - 10:
            break
        
        # Check if target is achievable
        current_radius = _compute_radius(x0)
        
        if target_r > current_radius * 1.15:
            # Target too aggressive, reduce it
            target_r = current_radius * 1.02
        elif target_r < current_radius * 0.98:
            # Target too conservative, increase it
            target_r = current_radius * 1.03
        
        # Run optimization
        try:
            result = least_squares(
                _residuals,
                x0.flatten(),
                args=(n, target_r),
                method='trf',
                max_nfev=5000,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
                verbose=0
            )
            
            new_centres = result.x.reshape(n, 2)
            
            # Validate
            norms = np.linalg.norm(new_centres, axis=1)
            if np.max(norms) > 1.0 + 1e-6:
                continue
            
            new_r = _compute_radius(new_centres)
            
            if new_r > best_r:
                best_r = new_r
                best_centres = new_centres
                x0 = new_centres
                current_r = new_r
                
                # Increase target for next iteration
                if new_r >= target_r * 0.995:
                    target_r = min(target_r * 1.03, 0.055)
                else:
                    target_r = new_r * 1.01
            else:
                # No improvement, try slightly different target
                target_r = max(target_r * 0.98, current_r * 1.005)
                
        except Exception:
            break
        
        if time.time() - start_time > max_time - 5:
            break
    
    # Final validation and return
    if best_centres is not None and best_r > 0:
        # Ensure all points are strictly inside
        norms = np.linalg.norm(best_centres, axis=1)
        if np.max(norms) > 1.0 - best_r + 1e-9:
            # Scale down slightly if needed
            scale = (1.0 - best_r - 1e-9) / np.max(norms)
            if scale < 1.0:
                best_centres = best_centres * scale
        
        return best_centres
    
    # Fallback: return hexagonal lattice
    try:
        centres = _hex_initial(n, 0.05)
        norms = np.linalg.norm(centres, axis=1)
        if np.max(norms) > 1.0:
            centres *= 0.999 / np.max(norms)
        return centres
    except Exception:
        # Last resort: uniform grid
        xs = np.linspace(-0.9, 0.9, 20)
        ys = np.linspace(-0.9, 0.9, 20)
        centres = np.array([[x, y] for x in xs for y in ys])[:n]
        norms = np.linalg.norm(centres, axis=1)
        if np.max(norms) > 1.0:
            centres *= 0.99 / np.max(norms)
        return centres

# EVOLVE-BLOCK-END
