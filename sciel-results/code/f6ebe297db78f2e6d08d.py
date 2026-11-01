import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(centers):
    """Compute the packing radius for a given set of centers."""
    # Centers shape (N, 3)
    # 1. Distance to walls
    # min_k (0.5 - |c_ik|)
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # 2. Pairwise distances
    # Use broadcasting for efficiency
    # diff[i, j] = centers[i] - centers[j]
    # This creates an (N, N, 3) array. For N=50, this is 50*50*3 = 7500 floats. Very small.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def _repulsive_potential(x):
    """
    Objective function to minimize.
    x: flat array of length 3*N
    Returns: sum of inverse distances between spheres and to walls.
    """
    centers = x.reshape(N, 3)
    
    # Wall distances
    # We want to penalize closeness to walls.
    # d_wall = 0.5 - |c|
    # Potential = 1 / d_wall
    # Clip to avoid division by zero or negative values if slightly out of bounds
    # Bounds are enforced by L-BFGS-B, but numerical errors can occur.
    wall_dists = 0.5 - np.abs(centers)
    # Clip to a small positive number
    wall_dists = np.clip(wall_dists, 1e-6, None)
    wall_pot = np.sum(1.0 / wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Clip distances
    dists = np.clip(dists, 1e-6, None)
    
    # Sum over i < j
    # Use triu_indices to get upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_pot = np.sum(1.0 / pair_dists)
    
    return wall_pot + pair_pot

def _grad_repulsive_potential(x):
    """
    Gradient of the repulsive potential.
    """
    centers = x.reshape(N, 3)
    grad = np.zeros_like(centers)
    
    # Wall gradient
    # d/dx (1 / (0.5 - |x|)) = 1 / (0.5 - |x|)^2 * sign(x)
    wall_dists = 0.5 - np.abs(centers)
    wall_dists = np.clip(wall_dists, 1e-6, None)
    signs = np.sign(centers)
    grad_wall = (1.0 / (wall_dists ** 2)) * signs
    grad += grad_wall
    
    # Pairwise gradient
    # Force on i from j: F_ij = -dU/dx_i = - (1/d^2) * (x_i - x_j) / d * (-1)?
    # U = 1/d. dU/dx_i = -1/d^2 * (x_i - x_j)/d = -(x_i - x_j)/d^3.
    # We want to minimize U, so we move in direction of -grad U.
    # grad U = sum_j -(x_i - x_j)/d^3.
    # So grad += -(x_i - x_j)/d^3 for all j != i.
    # Which is sum_j (x_j - x_i)/d^3.
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    dists = np.clip(dists, 1e-6, None)
    
    # term_ij = (x_j - x_i) / d_ij^3
    # diff is x_i - x_j. So x_j - x_i = -diff.
    # term = -diff / dists^3
    inv_d3 = 1.0 / (dists ** 3)
    # inv_d3 has inf on diagonal, we filled dists with inf, so inv_d3 is 0 on diagonal.
    # But we should be careful.
    # grad_pair[i] = sum_j (x_j - x_i) / d_ij^3 = - sum_j (x_i - x_j) / d_ij^3
    # = - sum_j diff[i,j] * inv_d3[i,j]
    
    # Sum over j for each i
    # grad_pair = - np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
    grad_pair = -np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
    grad += grad_pair
    
    return grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -np.inf
    best_centers = None
    
    # Initial BCC configuration
    # 3x3x3 cells, 2 points per cell = 54 points.
    # We take 50.
    # Coordinates in cell units [0, 2.5]
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append((i, j, k))
                pts.append((i + 0.5, j + 0.5, k + 0.5))
                if len(pts) >= 50:
                    break
            if len(pts) >= 50:
                break
        if len(pts) >= 50:
            break
    
    pts = np.array(pts[:50])
    
    # Map [0, 2.5] to [-0.5, 0.5]
    # Center is 1.25. Range is 2.5.
    # x_new = (x - 1.25) / 1.25 * 0.5
    centers = (pts - 1.25) / 1.25 * 0.5
    
    # Ensure valid
    centers = np.clip(centers, -0.499, 0.499)
    
    # Try a few restarts with different initializations
    # Restart 1: BCC
    # Restart 2: Jittered BCC
    # Restart 3: Random
    
    restarts = [
        centers.copy(),
        centers + np.random.normal(0, 0.01, centers.shape),
        np.random.uniform(-0.4, 0.4, (N, 3))
    ]
    
    for r_idx, init_centers in enumerate(restarts):
        if time.time() > deadline:
            break
            
        x0 = init_centers.flatten()
        
        # Use L-BFGS-B
        try:
            res = minimize(
                _repulsive_potential,
                x0,
                jac=_grad_repulsive_potential,
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * (3 * N),
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception as e:
            # If optimization fails, keep the initial guess
            res = None
        
        if res is not None and res.success:
            final_centers = res.x.reshape(N, 3)
            # Clip to ensure validity
            final_centers = np.clip(final_centers, -0.499, 0.499)
            
            score = _compute_radius(final_centers)
            
            if score > best_score:
                best_score = score
                best_centers = final_centers
        else:
            # If optimization didn't succeed or failed, evaluate initial
            score = _compute_radius(init_centers)
            if score > best_score:
                best_score = score
                best_centers = init_centers
        
        # Check time
        if time.time() > deadline - 20.0:
             # If we have less than 20s, stop trying new restarts
             break

    if best_centers is None:
        # Fallback to simple grid if everything failed
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
