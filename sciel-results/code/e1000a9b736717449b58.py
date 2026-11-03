"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_start_time = time.time()
_DEADLINE = 140.0  # Keep 10s safety margin for 150s limit

def _hex_lattice_init():
    """Generate initial points on a hexagonal lattice."""
    # Estimate spacing
    # Area of unit circle = pi. Area of hex cell = sqrt(3)/2 * s^2.
    # N ~ pi / (sqrt(3)/2 * s^2) => s ~ sqrt(2*pi / (N*sqrt(3)))
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Generate a grid of hex points
    # k is number of steps in each direction
    k = int(1.0 / s) + 3
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    radii = np.linalg.norm(points, axis=1)
    
    # Keep points inside the unit circle (with some margin for radius)
    # We want points that can potentially fit circles of radius ~s/2
    # So |p| + s/2 <= 1 => |p| <= 1 - s/2
    mask = radii <= 1.0 - s / 2.0
    points = points[mask]
    radii = radii[mask]
    
    if len(points) < N:
        # If not enough points, expand k or reduce s
        # For N=50, this shouldn't happen with reasonable s
        # But just in case, return random points? No, better to adjust s
        # Let's just return what we have, or pad with random if necessary
        # But for N=50, hex lattice is dense enough.
        pass

    # Sort by distance to center and take N closest
    idx = np.argsort(radii)
    return points[idx[:N]]

def _penalty_and_grad(x, r):
    """
    Calculate penalty for a configuration x given target radius r.
    x: (N, 2) array of centers.
    r: target radius.
    Returns: (cost, gradient)
    """
    x = np.asarray(x).reshape(N, 2)
    
    # 1. Circle-Circle Overlaps
    # d_ij = ||c_i - c_j||
    # Penalty P_ij = max(0, 2r - d_ij)^2
    # We want to minimize sum P_ij
    
    # Compute pairwise differences
    # diff[i, j] = c_i - c_j
    # Using broadcasting
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq)
    
    # Avoid division by zero in gradient
    dist_safe = np.where(dist < 1e-12, 1e-12, dist)
    
    # Overlap condition: dist < 2r
    overlap_mask = dist < 2.0 * r
    
    # Penalty terms
    # P_ij = (2r - dist)^2 if dist < 2r else 0
    # We only need to compute for i < j to avoid double counting and self-interaction
    # But vectorized calculation is easier with full matrix, then sum upper triangle
    
    # Gradient contribution from overlap
    # d(P_ij)/d(c_i) = 2(2r - dist) * (c_i - c_j) / dist
    # d(P_ij)/d(c_j) = 2(2r - dist) * (c_j - c_i) / dist
    
    # Initialize gradient
    grad = np.zeros_like(x)
    cost = 0.0
    
    if np.any(overlap_mask):
        # Compute penalty values
        # Only for i < j
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        mask_ij = overlap_mask[iu]
        
        if np.any(mask_ij):
            d_ij_valid = d_ij[mask_ij]
            diff_ij = diff[iu][mask_ij] # c_i - c_j
            
            penalty_vals = (2.0 * r - d_ij_valid) ** 2
            cost += np.sum(penalty_vals)
            
            # Gradient factor
            # factor = 2 * (2r - d) / d
            factors = 2.0 * (2.0 * r - d_ij_valid) / d_ij_valid
            
            # Accumulate gradient
            # For each pair (i, j):
            # grad[i] += factor * (c_i - c_j)
            # grad[j] -= factor * (c_i - c_j)  (since c_j - c_i = -(c_i - c_j))
            
            idx_i, idx_j = iu
            idx_i_valid = idx_i[mask_ij]
            idx_j_valid = idx_j[mask_ij]
            
            # Use np.add.at for accumulation
            np.add.at(grad, idx_i_valid, factors[:, np.newaxis] * diff_ij)
            np.add.at(grad, idx_j_valid, -factors[:, np.newaxis] * diff_ij)

    # 2. Boundary Violations
    # Q_i = max(0, r + |c_i| - 1)^2
    # Penalty if |c_i| > 1 - r
    
    radii = np.linalg.norm(x, axis=1)
    boundary_mask = radii > 1.0 - r
    
    if np.any(boundary_mask):
        radii_valid = radii[boundary_mask]
        x_valid = x[boundary_mask]
        
        penalty_vals = (r + radii_valid - 1.0) ** 2
        cost += np.sum(penalty_vals)
        
        # Gradient
        # d(Q_i)/d(c_i) = 2(r + |c_i| - 1) * c_i / |c_i|
        # factor = 2 * (r + r_i - 1) / r_i
        factors = 2.0 * (r + radii_valid - 1.0) / np.where(radii_valid < 1e-12, 1e-12, radii_valid)
        
        idx_valid = np.where(boundary_mask)[0]
        np.add.at(grad, idx_valid, factors[:, np.newaxis] * x_valid)

    return cost, grad

def _try_pack(r, x0, max_restarts=3):
    """
    Try to find a valid packing for radius r starting from x0.
    Returns (x_opt, success)
    """
    best_x = x0
    best_cost = np.inf
    
    for _ in range(max_restarts):
        if time.time() > _DEADLINE:
            break
            
        # Perturb initial position
        # Small perturbation to escape local minima
        x_perturbed = x0 + np.random.normal(0, 0.01, size=x0.shape)
        
        # Optimize
        try:
            res = minimize(
                _penalty_and_grad, 
                x_perturbed, 
                args=(r,), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x
                
            # Check if valid
            if res.fun < 1e-6:
                return best_x, True
                
        except Exception:
            continue
            
    return best_x, False

def _get_radius(x):
    """Calculate the actual radius of the packing."""
    x = np.asarray(x).reshape(N, 2)
    
    # Min distance between circles
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Zero out diagonal
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    
    # Min distance to boundary
    radii = np.linalg.norm(x, axis=1)
    min_border_dist = 1.0 - np.max(radii)
    
    r = min(min_dist / 2.0, min_border_dist)
    return max(0, r)

def construct():
    global _start_time
    
    # 1. Initialize
    x0 = _hex_lattice_init()
    current_r = _get_radius(x0)
    
    best_x = x0
    best_r = current_r
    
    # 2. Binary Search for optimal r
    # Lower bound is current_r
    # Upper bound: Theoretical max for N=50 is ~0.1258. 
    # Hex lattice gives ~0.125. Let's try up to 0.126.
    r_low = current_r
    r_high = 0.1260
    
    # If current_r is already high, adjust r_high
    if current_r > 0.1255:
        r_high = current_r + 0.001
    
    # Binary Search Loop
    for _ in range(40): # Max 40 iterations
        if time.time() > _DEADLINE:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to pack with radius r_mid
        x_opt, success = _try_pack(r_mid, best_x, max_restarts=2)
        
        if success:
            # Update best if this r_mid is better
            # The actual radius of x_opt might be slightly higher than r_mid
            actual_r = _get_radius(x_opt)
            if actual_r > best_r:
                best_r = actual_r
                best_x = x_opt
            
            r_low = r_mid
        else:
            r_high = r_mid
            
        # Convergence check
        if r_high - r_low < 1e-5:
            break
            
    # 3. Final Polish
    # Try to optimize the best_x found to maximize the minimum distance directly?
    # Or just ensure best_x is the best we found.
    # We can run one more optimization on best_x to minimize violations for a slightly larger r?
    # No, the binary search already maximized the feasible r.
    # However, the last successful x_opt was for r_low.
    # We might have a better x from the last "failed" attempt? No, failed means invalid.
    
    # Return the best configuration found
    return best_x
# EVOLVE-BLOCK-END
