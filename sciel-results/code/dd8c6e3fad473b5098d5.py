import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(x):
    """
    x: shape (N, 3)
    Returns energy and gradient.
    Energy: Sum of inverse square distances between all pairs + inverse square distances to walls.
    """
    # Reshape to (N, 3)
    x = x.reshape(N, 3)
    
    # Compute pairwise distances
    # x[:, np.newaxis, :] - x[np.newaxis, :, :] gives (N, N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dist_sq_masked = np.where(mask, dist_sq, 1.0)
    
    # Energy from pairs: sum 1/d^2
    # We only want i < j to avoid double counting, but summing all i!=j is fine for gradient, just 2x.
    # Let's sum over i < j to be precise.
    # Actually, summing all i!=j is easier with matrix ops.
    # E_pairs = sum_{i!=j} 1/d_ij^2
    # But this is 2 * sum_{i<j} 1/d_ij^2
    # Let's just use the full matrix sum, it's proportional.
    e_pairs = np.sum(1.0 / dist_sq_masked)
    
    # Gradient of E_pairs w.r.t x_i
    # d/dx_i (1/|x_i - x_j|^2) = -2 (x_i - x_j) / |x_i - x_j|^4
    # Sum over j
    # Let's compute gradient numerically or analytically.
    # Analytical:
    # grad_i = sum_{j!=i} -2 * (x_i - x_j) / dist_ij^4
    # dist_sq is (N, N)
    # dist_sq_masked is (N, N)
    # We need dist_sq^2 for the denominator? No, dist_sq is d^2. So d^4 is dist_sq^2.
    # Factor = 1 / (dist_sq_masked**2)
    factor = 1.0 / (dist_sq_masked**2)
    # Set diagonal to 0
    factor[np.eye(N, dtype=bool)] = 0.0
    
    # grad_pairs = -2 * sum_j (x_i - x_j) * factor[i, j]
    # (x_i - x_j) is diff[i, j]
    # We need sum over j of diff[i, j] * factor[i, j]
    # diff is (N, N, 3), factor is (N, N)
    # Multiply factor[:, :, np.newaxis] with diff
    weighted_diff = diff * factor[:, :, np.newaxis]
    grad_pairs = -2.0 * np.sum(weighted_diff, axis=1)
    
    # Wall distances
    # x is in [-0.5, 0.5]
    # Distance to left wall: x + 0.5
    # Distance to right wall: 0.5 - x
    # d_left = x + 0.5
    # d_right = 0.5 - x
    # E_walls = sum_i sum_k (1/d_left^2 + 1/d_right^2)
    # To avoid division by zero if x is exactly 0.5 or -0.5, we add a small epsilon or rely on bounds.
    # Since we use minimize with bounds, x will be strictly inside or on boundary.
    # If on boundary, d=0, energy=inf. This is good for optimization, it pushes in.
    
    d_left = x + 0.5
    d_right = 0.5 - x
    
    # Add small epsilon to prevent inf/nan if exactly on wall
    eps = 1e-6
    d_left = np.maximum(d_left, eps)
    d_right = np.maximum(d_right, eps)
    
    e_walls = np.sum(1.0 / (d_left**2)) + np.sum(1.0 / (d_right**2))
    
    # Gradient of E_walls
    # d/dx (1/(x+0.5)^2) = -2 / (x+0.5)^3
    # d/dx (1/(0.5-x)^2) = 2 / (0.5-x)^3
    grad_walls = -2.0 / (d_left**3) + 2.0 / (d_right**3)
    
    total_energy = e_pairs + e_walls
    total_grad = grad_pairs + grad_walls
    
    return total_energy, total_grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Initial guess: Jittered grid
    # 5x4x5 = 100 points
    # Grid coordinates in [-0.5, 0.5]
    # Spacing: 1/5 = 0.2 in x, 1/4 = 0.25 in y, 1/5 = 0.2 in z
    # Centers: -0.4, -0.2, 0, 0.2, 0.4 for x/z
    #           -0.375, -0.125, 0.125, 0.375 for y
    
    x_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    y_ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    z_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    
    # Create grid
    grid = np.array([(x, y, z) for x in x_ticks for y in y_ticks for z in z_ticks])
    
    # Add jitter to break symmetry
    np.random.seed(42)
    jitter = np.random.uniform(-0.02, 0.02, grid.shape)
    x0 = grid + jitter
    
    # Ensure initial points are within bounds
    x0 = np.clip(x0, -0.49, 0.49)
    
    best_x = x0
    best_score = -1.0
    
    # Define objective function for scipy
    def objective(x_flat):
        e, g = _energy_and_grad(x_flat)
        return e, g
        
    # Optimize
    # Use L-BFGS-B
    # Maxiter: limit iterations or time
    # Since we can't easily check time in callback, we limit maxiter.
    # 300 variables. L-BFGS-B is fast.
    # Let's try 1000 iterations.
    
    try:
        res = minimize(objective, x0.flatten(), method='L-BFGS-B', jac=True,
                       bounds=[(-0.5, 0.5)] * 3 * N,
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.success:
            best_x = res.x.reshape(N, 3)
            
    except Exception as e:
        print("Optimization failed:", e)
        
    # Calculate score
    # Score is min(min_pair_dist/2, min_wall_dist)
    def calc_score(pts):
        # Pair distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1e10)
        min_pair_dist = np.sqrt(np.min(dist_sq))
        
        # Wall distances
        min_wall_dist = np.min(0.5 - np.abs(pts))
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    current_score = calc_score(best_x)
    
    # If we have time, we can try a few random restarts or a second phase of optimization
    # But L-BFGS-B from a good start is usually sufficient.
    # Let's check time.
    if time.time() < deadline - 20:
        # Try a second optimization with a different initial guess or more iterations
        # Or just refine the current one
        try:
            res2 = minimize(objective, best_x.flatten(), method='L-BFGS-B', jac=True,
                            bounds=[(-0.5, 0.5)] * 3 * N,
                            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
            if res2.success:
                new_x = res2.x.reshape(N, 3)
                new_score = calc_score(new_x)
                if new_score > current_score:
                    best_x = new_x
                    current_score = new_score
        except Exception:
            pass
            
    return best_x

# EVOLVE-BLOCK-END
