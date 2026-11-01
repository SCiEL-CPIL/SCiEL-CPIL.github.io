import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_energy_and_grad(x_flat, r):
    """
    x_flat: array of shape (2*N,) containing [x1, y1, x2, y2, ...]
    r: scalar radius
    Returns: energy, gradient
    """
    x = x_flat.reshape(N, 2)
    energy = 0.0
    grad = np.zeros_like(x)
    
    # Inter-circle overlaps
    # d_ij^2 = (x_i - x_j)^2 + (y_i - y_j)^2
    # We want d_ij >= 2r
    # Penalty: max(0, 2r - d_ij)^2
    # Gradient wrt x_i: 2 * max(0, 2r - d_ij) * (x_i - x_j) / d_ij
    for i in range(N):
        for j in range(i + 1, N):
            diff = x[i] - x[j]
            d_sq = np.dot(diff, diff)
            if d_sq < 1e-12:
                # Coincident points, handle carefully to avoid division by zero
                # Push them apart slightly
                d = 1e-6
                # This case is rare with random init, but possible
                # We'll add a small random kick in the gradient
                grad[i] += np.random.normal(0, 0.1)
                grad[j] -= np.random.normal(0, 0.1)
                energy += (2*r)**2 # Approx penalty
                continue
            d = np.sqrt(d_sq)
            if d < 2*r:
                # Overlap
                violation = 2*r - d
                energy += violation * violation
                # Gradient of (2r - d)^2 wrt x_i
                # d/dx_i (2r - d)^2 = 2(2r-d) * d/dx_i (2r-d) = 2(2r-d) * (-1) * d/dx_i d
                # d/dx_i d = (x_i - x_j) / d
                # So grad_i = -2(2r-d) * (x_i - x_j) / d
                # Wait, we want to minimize energy.
                # If d < 2r, we want to increase d.
                # Force on i should be away from j.
                # Vector from j to i is (x_i - x_j).
                # So grad should be in direction (x_i - x_j).
                # My derivative:
                # f = (2r - d)^2
                # df/dx_i = 2(2r - d) * (-1) * (x_i - x_j)/d
                # Since 2r - d > 0, and (x_i - x_j)/d is unit vector from j to i.
                # The negative sign means the gradient points towards j?
                # No, minimize f. If we move x_i in direction of gradient, f increases.
                # We want to move in direction -gradient.
                # -gradient = 2(2r-d) * (x_i - x_j)/d. This points away from j. Correct.
                coeff = 2 * (2*r - d) / d
                grad[i] += coeff * diff
                grad[j] -= coeff * diff
                
    # Wall constraints
    # x_i <= 0.5 - r  => 0.5 - r - x_i >= 0
    # y_i <= 0.5 - r
    # -x_i <= 0.5 - r => x_i >= -0.5 + r
    # -y_i <= 0.5 - r
    # Penalty: max(0, x_i - (0.5 - r))^2 + max(0, -x_i - (-0.5 + r))^2
    #          max(0, y_i - (0.5 - r))^2 + max(0, -y_i - (-0.5 + r))^2
    bound = 0.5 - r
    for i in range(N):
        for k in range(2):
            val = x[i, k]
            # Right/Top wall
            if val > bound:
                v = val - bound
                energy += v * v
                grad[i, k] += 2 * v
            # Left/Bottom wall
            if val < -bound:
                v = -bound - val
                energy += v * v
                # d/dval (-bound - val)^2 = 2(-bound - val)(-1) = 2(bound + val)
                # Wait.
                # f = (-bound - val)^2.
                # df/dval = 2(-bound - val) * (-1) = 2(bound + val).
                # If val < -bound, bound + val < 0.
                # So gradient is negative.
                # -gradient is positive. Move val up. Correct.
                grad[i, k] += 2 * (bound + val)
                
    return energy, grad.flatten()

def optimize_positions(r, x0, time_limit=10.0):
    """
    Minimize energy for fixed r.
    x0: initial positions (N, 2)
    """
    x0_flat = x0.flatten()
    
    # L-BFGS-B is good for this
    res = minimize(
        fun=lambda x: compute_energy_and_grad(x, r)[0],
        x0=x0_flat,
        jac=lambda x: compute_energy_and_grad(x, r)[1],
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (2 * N),
        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    return res.x.reshape(N, 2), res.fun

def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    # Initial grid
    m = 7  # 7x7 grid has 49, we need 50. 8x8 has 64.
    # Let's use a 7x7 grid plus one point, or just 8x8 and take 50.
    # 8x8 grid: spacing 1/8 = 0.125. Radius 0.0625.
    # 7x7 grid: spacing 1/7 ~ 0.142. Radius 0.0714.
    # 50 points in 7x7: 49 fit perfectly. The 50th must be added, likely causing overlap or forcing shrinkage.
    # Actually, 7x7 = 49. We need 50.
    # So 7x7 grid is a good start. Place 49 points, and the 50th in the center of a hole?
    # Or just use 8x8 grid and pick 50 points.
    # Let's start with a 7x7 grid for 49 points, and add the 50th at (0,0) or somewhere.
    # Better: Start with a random perturbation of a 7x7 grid + 1 point.
    
    # Create 7x7 grid
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    grid_49 = np.array([(x, y) for y in ticks for x in ticks])
    
    # Add 50th point. Center (0,0) is occupied if 7 is odd.
    # 7x7 grid centers: -3/7, -2/7, -1/7, 0, 1/7, 2/7, 3/7.
    # (0,0) is the center.
    # Let's place the 50th point at (0.5/7, 0) or just perturb everything.
    # Simplest: 8x8 grid, take first 50.
    ticks_8 = (np.arange(8) + 0.5) / 8 - 0.5
    grid_64 = np.array([(x, y) for y in ticks_8 for x in ticks_8])
    x0 = grid_64[:50].copy()
    
    # Add small random perturbation to break symmetry
    x0 += np.random.normal(0, 0.001, x0.shape)
    
    # Binary search for r
    r_low = 0.0700
    r_high = 0.0714
    best_x = x0
    best_r = r_low
    best_score = 0.0
    
    def get_score(x):
        d_min = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(x[i] - x[j])
                if d < d_min:
                    d_min = d
        wall_min = np.inf
        for i in range(N):
            for k in range(2):
                d_wall = 0.5 - abs(x[i, k])
                if d_wall < wall_min:
                    wall_min = d_wall
        return min(d_min / 2, wall_min)
    
    # Initial score
    best_score = get_score(best_x)
    
    for i_iter in range(30):
        if time.time() - start_time > deadline:
            break
            
        r_mid = (r_low + r_high) / 2
        x_opt, energy = optimize_positions(r_mid, best_x)
        
        # Check if feasible
        # Energy should be close to 0
        if energy < 1e-5:
            # Feasible
            r_low = r_mid
            best_x = x_opt
            current_score = get_score(best_x)
            if current_score > best_score:
                best_score = current_score
        else:
            r_high = r_mid
            
    # Final polish: Try to optimize for the best_r found?
    # The binary search converges to the max r.
    # The returned best_x corresponds to r_low (the last feasible one).
    
    # One more optimization step on best_x to clean up
    # Use the best_score as the target r for a final polish?
    # No, just return best_x.
    
    return best_x

# EVOLVE-BLOCK-END
