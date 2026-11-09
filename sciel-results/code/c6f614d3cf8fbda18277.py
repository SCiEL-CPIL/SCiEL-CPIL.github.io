import numpy as np
import time
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # 1. Generate initial hexagonal packing
    # Approximate number of circles per row for hex packing in unit square
    # Side length 1. Hex packing density ~ 0.9069.
    # Area of N circles = N * pi * r^2.
    # If packed in square, N * pi * r^2 <= 1.
    # r ~ 1/sqrt(N*pi) ~ 1/sqrt(628) ~ 0.04.
    # Hex spacing: horizontal dx = 2r, vertical dy = sqrt(3)r.
    # Let's just generate a grid of hex points and pick the best N.
    
    # A simpler robust init: 
    # Generate points on a hex lattice covering a slightly larger area, 
    # then clip to square, then if too many, remove random ones or keep best N.
    # Or just use a square grid and jitter? 
    # Hex is better.
    
    r_est = 0.037 # Known best is ~0.0366
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    # Generate x, y ranges
    x_vals = np.arange(-0.5, 0.5 + dx, dx)
    y_vals = np.arange(-0.5, 0.5 + dy, dy)
    
    # Create hex grid
    points = []
    for i, y in enumerate(y_vals):
        # Offset alternate rows
        offset = 0 if i % 2 == 0 else dx / 2
        for x in x_vals:
            px = x + offset
            if -0.5 <= px <= 0.5 and -0.5 <= y <= 0.5:
                points.append([px, y])
    
    points = np.array(points)
    
    # If we have too many, we need to reduce to N.
    # If we have too few, we need to add.
    # With r=0.037, area covered is approx 200 * pi * 0.037^2 ~ 0.86.
    # Square area is 1. So we should have around 200 points.
    
    if len(points) > N:
        # Keep points closest to the center? Or random?
        # Random selection from a good initial structure is often fine, 
        # but keeping a connected subset might be better.
        # Let's just take the first N after shuffling to break symmetry?
        # Or better: take the N points that are "most central" to avoid boundary issues initially?
        # Actually, for optimization, it's better to have a roughly uniform distribution.
        # Let's just take a random subset of size N.
        idx = np.random.choice(len(points), N, replace=False)
        x0 = points[idx]
    elif len(points) < N:
        # Need to add points. This is tricky.
        # Let's restart with a slightly smaller r_est to get more points.
        # Or just use a square grid of size ceil(sqrt(N)) which guarantees N points.
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        x0 = grid[:N]
        # Add small random jitter to break symmetry
        x0 += np.random.normal(0, 0.001, x0.shape)
    else:
        x0 = points

    # Ensure x0 is float and shape (N, 2)
    x0 = np.array(x0, dtype=float)
    x0 = np.clip(x0, -0.5, 0.5)
    
    # Define objective and gradient
    beta = 30.0 # Soft-min parameter
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 2)
        
        # Pairwise distances
        # diff[i, j] = x[i] - x[j]
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero
        dist_sq[dist_sq == 0] = 1e-12
        dist = np.sqrt(dist_sq)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dist = dist[mask]
        diff_flat = diff[mask] # Shape (N*(N-1), 2)
        
        # Boundary distances
        # d_left = 0.5 + x
        # d_right = 0.5 - x
        # d_bottom = 0.5 + y
        # d_top = 0.5 - y
        d_bound = np.concatenate([
            0.5 + x[:, 0],
            0.5 - x[:, 0],
            0.5 + x[:, 1],
            0.5 - x[:, 1]
        ])
        
        # Combine all distances
        all_dist = np.concatenate([dist, d_bound])
        
        # Soft min calculation
        # exp(-beta * dist)
        exp_term = np.exp(-beta * all_dist)
        S = np.sum(exp_term)
        
        # Objective: -1/beta * ln(S)
        # We want to MAXIMIZE this, so we minimize -objective
        obj_val = -(-1.0 / beta * np.log(S)) # This is just 1/beta * ln(S)? 
        # Wait. 
        # f_soft = -1/beta ln(sum exp(-beta d))
        # If d_min is small, sum is dominated by exp(-beta d_min).
        # f_soft ~ -1/beta (-beta d_min) = d_min.
        # So f_soft approximates d_min.
        # We want to maximize d_min.
        # So we maximize f_soft.
        # scipy.optimize.minimize minimizes.
        # So we minimize -f_soft.
        
        # Gradient of f_soft
        # grad f_soft = 1/S * sum (exp(-beta d_k) * grad d_k)
        # Note: grad d_k is with respect to x.
        
        grad = np.zeros_like(x)
        
        # Pairwise contribution
        # For each pair (i, j) with i < j, dist is calculated.
        # In diff_flat, the vector is x_i - x_j.
        # grad_{x_i} d_ij = (x_i - x_j) / d_ij
        # grad_{x_j} d_ij = (x_j - x_i) / d_ij = - (x_i - x_j) / d_ij
        
        # We have diff_flat corresponding to dist.
        # The indices in diff_flat are from mask.
        # We need to map back to i, j.
        # This is slow if done naively. 
        # Let's use the full matrix approach for gradient accumulation.
        
        # Let's reconstruct the full matrix gradient contributions.
        # W_ij = exp(-beta * dist_ij) / S
        # For i != j:
        #   grad_xi += W_ij * (x_i - x_j) / dist_ij
        #   grad_xj += W_ij * (x_j - x_i) / dist_ij
        
        # We can compute this using the full NxN matrices.
        # exp_matrix = exp(-beta * dist_matrix)
        # Set diagonal to 0.
        # S = sum(exp_matrix)
        
        # Let's redo the calculation using full matrices for clarity and speed in NumPy.
        
        diff_mat = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff_mat**2, axis=2) + 1e-12)
        
        # Mask diagonal
        np.fill_diagonal(dist_mat, 0.0)
        
        exp_mat = np.exp(-beta * dist_mat)
        np.fill_diagonal(exp_mat, 0.0)
        
        S_pair = np.sum(exp_mat)
        
        # Total S
        # Boundary exp terms
        exp_bound = np.exp(-beta * d_bound)
        S_bound = np.sum(exp_bound)
        
        S_total = S_pair + S_bound
        
        # Objective
        f_soft = -1.0 / beta * np.log(S_total)
        # Minimize -f_soft
        obj = -f_soft
        
        # Gradient
        # grad f_soft = (1/S_total) * (Sum_pair + Sum_bound)
        
        # Pairwise part:
        # Term for (i,j) is exp_mat[i,j] * (x_i - x_j) / dist_mat[i,j]
        # Let V = exp_mat / dist_mat (with 0 on diagonal)
        # Contribution to grad_xi from j is V[i,j] * (x_i - x_j)
        # Sum over j: V[i,:] @ x - (V[i,:] * x_i) ? No.
        # Sum_j V[i,j] (x_i - x_j) = x_i * Sum_j V[i,j] - Sum_j V[i,j] x_j
        
        V = exp_mat / dist_mat
        # Handle division by zero on diagonal (already 0/0 -> nan, set to 0)
        V[~np.isfinite(V)] = 0.0
        
        row_sums = np.sum(V, axis=1) # Shape (N,)
        # Sum_j V[i,j] x_j is V @ x
        Vx = V @ x # Shape (N, 2)
        
        grad_pair = row_sums[:, np.newaxis] * x - Vx
        
        # Boundary part:
        # d_bound is concatenated: 
        # 0:N: 0.5 + x_i (Left)
        # N:2N: 0.5 - x_i (Right)
        # 2N:3N: 0.5 + y_i (Bottom)
        # 3N:4N: 0.5 - y_i (Top)
        
        w_bound = exp_bound / S_total
        
        # Grad for x component:
        # Left: d = 0.5 + x_i, grad_x = 1
        # Right: d = 0.5 - x_i, grad_x = -1
        # Bottom: d = 0.5 + y_i, grad_y = 1
        # Top: d = 0.5 - y_i, grad_y = -1
        
        w_left = w_bound[0:N]
        w_right = w_bound[N:2*N]
        w_bottom = w_bound[2*N:3*N]
        w_top = w_bound[3*N:4*N]
        
        grad_bound_x = w_left - w_right
        grad_bound_y = w_bottom - w_top
        
        grad_bound = np.column_stack((grad_bound_x, grad_bound_y))
        
        # Total gradient of f_soft
        grad_f_soft = (grad_pair + grad_bound) / S_total
        
        # We are minimizing -f_soft, so gradient is -grad_f_soft
        grad_obj = -grad_f_soft
        
        return obj, grad_obj

    # Run optimization
    # L-BFGS-B
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_x = x0.flatten()
    best_obj = np.inf
    
    # We can run multiple steps or just one call with maxiter
    # Let's use a loop to check time
    max_iter = 200
    current_x = x0.flatten()
    
    for _ in range(10): # Max 10 restarts/iterations if time permits
        if time.time() > deadline - 10.0:
            break
            
        res = minimize(
            objective_and_grad, 
            current_x, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-9, 'disp': False}
        )
        
        if res.success or res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x
        
        # If converged, break
        if res.status == 0:
            break
            
        # If time is running out, break
        if time.time() > deadline - 20.0:
            break
            
        # Perturb for next restart if not converged
        # Small perturbation to escape local minima
        current_x = best_x + np.random.normal(0, 0.005, best_x.shape)
        current_x = np.clip(current_x, -0.5, 0.5)
        
    # Final check
    if time.time() > deadline - 5.0:
         # Return best found
         pass
         
    final_x = best_x.reshape(N, 2)
    
    # Ensure validity
    final_x = np.clip(final_x, -0.5, 0.5)
    
    return final_x
# EVOLVE-BLOCK-END
