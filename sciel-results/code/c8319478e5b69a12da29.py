import numpy as np
import time
from scipy.optimize import minimize

N = 100
START_TIME = time.time()
TIME_LIMIT = 150.0 # Safety margin for evaluator

def construct():
    # 1. Initialization
    # Use a grid initialization to start close to a good solution
    # 100 spheres. 5x5x4 = 100.
    # Side length 1.
    # Grid points in [-0.5, 0.5]
    
    # We can try a few different grid shapes or just standard 5x5x4
    # Let's use 5x5x4
    nx, ny, nz = 5, 5, 4
    if nx * ny * nz < N:
        # Fallback to more generic
        m = int(np.ceil(N ** (1.0/3.0)))
        nx, ny, nz = m, m, m
        # This might be > N, we will truncate later
        
    # Create grid
    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    z = np.linspace(-0.5, 0.5, nz)
    
    # Generate all points
    grid_points = []
    for xi in x:
        for yi in y:
            for zi in z:
                grid_points.append([xi, yi, zi])
                
    grid_points = np.array(grid_points[:N])
    
    # Add small random noise to break symmetry
    # The grid is symmetric, leading to flat directions in the potential landscape.
    # Noise helps the optimizer find the unique global minimum (or a better local one).
    rng = np.random.default_rng(42)
    # Scale noise small enough not to violate constraints immediately
    noise = rng.normal(0, 0.01, size=(N, 3))
    centers = grid_points + noise
    
    # Ensure centers are strictly inside the cube
    # Clip to [-0.5 + eps, 0.5 - eps]
    eps = 1e-6
    centers = np.clip(centers, -0.5 + eps, 0.5 - eps)
    
    # 2. Define Objective and Gradient
    beta = 50.0 # Softness parameter. Higher = closer to hard min.
    
    def objective_and_grad(flat_centers):
        c = flat_centers.reshape(N, 3)
        
        # Compute pairwise distances / 2
        # diff[i, j] = c[i] - c[j]
        # dist[i, j] = ||c[i] - c[j]||
        # We only need i < j
        # Vectorized calculation
        # c is (N, 3)
        # dist_sq = sum((c[:, None, :] - c[None, :, :])**2, axis=2)
        # This creates an N x N matrix.
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Distances are sqrt(dist_sq) / 2
        # We need to be careful with i=j (dist=0). 
        # In the soft-min, we should exclude i=j.
        # Let's create a mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Extract distances
        # dist_sq_masked = dist_sq[mask]
        # dist_masked = np.sqrt(dist_sq_masked) / 2
        
        # Boundary distances
        # d_k = 0.5 - |c_ik|
        # For each center c_i, dist to boundary is min(0.5 - c_ix, 0.5 - c_iy, 0.5 - c_iz, 
        #                                              0.5 + c_ix, 0.5 + c_iy, 0.5 + c_iz)
        # Wait, the cube is [-0.5, 0.5].
        # Distance to x=0.5 is 0.5 - c_x. Distance to x=-0.5 is c_x - (-0.5) = c_x + 0.5.
        # So min_dist_to_boundary_i = min(0.5 - c_ix, 0.5 - c_iy, 0.5 - c_iz, 0.5 + c_ix, 0.5 + c_iy, 0.5 + c_iz)
        
        # Let's compute boundary distances for all i
        # b[i] = min(0.5 - |c[i,0]|, 0.5 - |c[i,1]|, 0.5 - |c[i,2]|)
        # Wait, min(0.5 - c, 0.5 + c) is NOT 0.5 - |c|.
        # If c=0.4, min(0.1, 0.9) = 0.1. 0.5 - |0.4| = 0.1.
        # If c=-0.4, min(0.9, 0.1) = 0.1. 0.5 - |-0.4| = 0.1.
        # Yes, min(0.5 - c, 0.5 + c) = 0.5 - |c|.
        
        abs_c = np.abs(c)
        # dist to boundary in x, y, z
        bx = 0.5 - abs_c[:, 0]
        by = 0.5 - abs_c[:, 1]
        bz = 0.5 - abs_c[:, 2]
        
        # Min over axes for each sphere
        b_dist = np.min(np.stack([bx, by, bz], axis=1), axis=1)
        
        # Combine pairwise and boundary distances into one vector D
        # D contains all constraints. We want to maximize the minimum of D.
        
        # Get pairwise distances
        # dist_sq is N x N. 
        # We can compute gradients analytically.
        
        # Let's compute the soft-min value and gradient.
        # D_vec = [dist_pairwise, b_dist]
        
        # To avoid creating huge temporary arrays, we can compute the gradient in parts.
        # But N=100, so N^2 = 10,000. This is small.
        
        # 1. Pairwise part
        # dist = sqrt(dist_sq) / 2
        # grad dist wrt c_i:
        # dist_ij = 0.5 * sqrt((ci-cj)^2)
        # d(dist_ij)/d(ci) = 0.5 * (ci-cj) / ||ci-cj|| = 0.5 * diff_ij / (2 * dist_ij) = diff_ij / (4 * dist_ij) ??
        # Let r_ij = ||ci-cj||. dist = r_ij / 2.
        # d(r_ij)/d(ci) = (ci-cj)/r_ij.
        # d(dist)/d(ci) = 0.5 * (ci-cj)/r_ij.
        
        # We need sum over j of w_ij * d(dist_ij)/d(ci)
        # w_ij = exp(-beta * dist_ij) / Sum_k exp(-beta * dist_k)
        
        # Let's compute all distances first
        # dist_sq has zeros on diagonal.
        # Set diagonal to a large number so exp(-beta * large) is 0?
        # Or just mask them out in the sum.
        
        # Let's compute the weights for all constraints.
        # D_pair = sqrt(dist_sq) / 2
        # D_bound = b_dist
        
        # We need to handle the case where dist is 0 (coincident points).
        # If dist is 0, gradient is undefined. 
        # With noise, this is unlikely.
        
        # To be safe, add a small epsilon to dist_sq before sqrt?
        # dist_sq_safe = dist_sq + 1e-12
        # dist = sqrt(dist_sq_safe) / 2
        
        # However, for the gradient, if dist is 0, the direction is arbitrary.
        # The noise prevents this.
        
        dist_sq_safe = dist_sq + 1e-12
        dist_pair = np.sqrt(dist_sq_safe) / 2.0
        
        # Calculate exponentials
        # To avoid underflow/overflow, we can shift by max? 
        # But beta is small (50), dist is ~0.1. beta*dist ~ 5. exp(-5) is fine.
        
        exp_pair = np.exp(-beta * dist_pair)
        exp_bound = np.exp(-beta * b_dist)
        
        # Sum of exponentials
        # Sum over i<j for pairs.
        # Sum over i for bounds.
        
        # Mask for i<j
        # exp_pair has values for all i,j. We only want i<j.
        # Since dist_pair is symmetric, exp_pair is symmetric.
        # Sum_{i<j} exp(-beta * dist_ij) = 0.5 * (Sum_{i!=j} exp(...) - Sum_{i=j} exp(...))
        # Sum_{i=j} exp(0) = N.
        # Sum_{i!=j} is sum of whole matrix minus diagonal.
        
        sum_exp_pair = 0.5 * (np.sum(exp_pair) - N)
        sum_exp_bound = np.sum(exp_bound)
        
        total_sum = sum_exp_pair + sum_exp_bound
        
        # Weights
        # w_pair_ij = exp_pair_ij / total_sum (for i<j)
        # w_bound_i = exp_bound_i / total_sum
        
        # Gradient of -softmin = - sum w_k * grad(D_k)
        # grad(softmin) = sum w_k * grad(D_k)
        # So we want to minimize -softmin, so we return -grad(softmin) as the gradient for the minimizer.
        # Wait, scipy minimize minimizes f. 
        # f = -softmin(D).
        # grad f = - grad softmin = - sum w_k grad D_k.
        
        grad_c = np.zeros((N, 3))
        
        # 1. Boundary contribution
        # D_bound_i = min(bx, by, bz)
        # This is non-differentiable at the min.
        # However, the soft-min over the *set* of all boundary distances (bx, by, bz for each i) 
        # is smoother than taking the min first.
        # Let's refine the definition of D.
        # D should be the set of all individual constraints:
        # 1. dist(c_i, c_j) / 2 for all i < j
        # 2. 0.5 - c_ix, 0.5 + c_ix, 0.5 - c_iy, 0.5 + c_iy, 0.5 - c_iz, 0.5 + c_iz for all i
        # This makes the soft-min smoother and handles the "min over axes" naturally.
        
        # Let's restart the calculation with this finer granularity.
        
        # Boundary terms:
        # For each i, we have 6 terms.
        # b_terms[i, 0] = 0.5 - c_i[0]
        # b_terms[i, 1] = 0.5 + c_i[0]
        # ...
        
        # Let's compute these explicitly.
        # b_terms shape (N, 6)
        b_terms = np.zeros((N, 6))
        b_terms[:, 0] = 0.5 - c[:, 0]
        b_terms[:, 1] = 0.5 + c[:, 0]
        b_terms[:, 2] = 0.5 - c[:, 1]
        b_terms[:, 3] = 0.5 + c[:, 1]
        b_terms[:, 4] = 0.5 - c[:, 2]
        b_terms[:, 5] = 0.5 + c[:, 2]
        
        exp_b_terms = np.exp(-beta * b_terms)
        sum_exp_b_terms = np.sum(exp_b_terms)
        
        # Pairwise terms
        # We already have exp_pair (N x N).
        # We need sum over i<j.
        # sum_exp_pair = 0.5 * (np.sum(exp_pair) - N)
        
        total_sum = sum_exp_pair + sum_exp_b_terms
        
        # Weights
        # w_pair_ij = exp_pair_ij / total_sum
        # w_b_term_k = exp_b_term_k / total_sum
        
        # Gradient Accumulation
        
        # Boundary Gradient
        # d(0.5 - c_ix)/d(c_ix) = -1
        # d(0.5 + c_ix)/d(c_ix) = 1
        # etc.
        
        # grad_c from boundary terms:
        # For each i:
        # contrib_x = w_b_term[i,0] * (-1) + w_b_term[i,1] * (1)
        # contrib_y = w_b_term[i,2] * (-1) + w_b_term[i,3] * (1)
        # contrib_z = w_b_term[i,4] * (-1) + w_b_term[i,5] * (1)
        
        w_b = exp_b_terms / total_sum
        
        grad_c[:, 0] += -w_b[:, 0] + w_b[:, 1]
        grad_c[:, 1] += -w_b[:, 2] + w_b[:, 3]
        grad_c[:, 2] += -w_b[:, 4] + w_b[:, 5]
        
        # Pairwise Gradient
        # d(dist_ij)/d(c_i) = 0.5 * (c_i - c_j) / r_ij
        # where r_ij = ||c_i - c_j||
        
        # We have diff = c[:, None, :] - c[None, :, :]
        # r_ij = sqrt(dist_sq_safe)
        # grad_dist_ij = 0.5 * diff / r_ij[:, :, None]
        
        # Note: diff[i, j] = c_i - c_j.
        # d(dist_ij)/d(c_i) = 0.5 * (c_i - c_j) / r_ij
        # d(dist_ij)/d(c_j) = 0.5 * (c_j - c_i) / r_ij = - d(dist_ij)/d(c_i)
        
        # We want sum_{j != i} w_ij * d(dist_ij)/d(c_i)
        # w_ij = exp_pair[i, j] / total_sum
        
        # grad_pair = sum_j w_ij * 0.5 * diff[i, j] / r_ij
        # = 0.5 * sum_j (exp_pair[i, j] / total_sum) * diff[i, j] / r_ij
        
        # Let's compute this vectorized.
        # r_ij is N x N.
        # diff is N x N x 3.
        # We need to divide diff by r_ij (broadcasting).
        
        # Handle r_ij = 0 (diagonal).
        # r_safe = np.where(r_ij > 1e-6, r_ij, 1.0)
        
        r_ij = np.sqrt(dist_sq_safe)
        # Avoid division by zero on diagonal
        r_safe = np.where(r_ij > 1e-6, r_ij, 1.0)
        
        # factor = exp_pair / (total_sum * 2 * r_safe)
        # We want 0.5 * exp_pair / (total_sum * r_safe)
        
        factor = exp_pair / (2.0 * total_sum * r_safe)
        
        # Zero out the diagonal in factor
        np.fill_diagonal(factor, 0.0)
        
        # grad_pair[i] = sum_j factor[i, j] * diff[i, j]
        # diff[i, j] is a vector of size 3.
        # factor[i, j] is a scalar.
        
        # grad_pair = np.sum(factor[:, :, None] * diff, axis=1)
        
        grad_pair = np.sum(factor[:, :, None] * diff, axis=1)
        
        grad_c += grad_pair
        
        # Total gradient of softmin is grad_c.
        # We want to minimize -softmin.
        # So gradient to return is -grad_c.
        
        # Calculate objective value: -softmin
        # softmin = -1/beta * log(total_sum)
        # f = -softmin = 1/beta * log(total_sum)
        
        f_val = (1.0 / beta) * np.log(total_sum)
        
        # Return f_val and gradient
        return f_val, -grad_c.flatten()
    
    # 3. Optimization
    # Use L-BFGS-B or CG
    # L-BFGS-B is good for box constraints, but we are using soft constraints.
    # We can use 'L-BFGS-B' with bounds to keep points inside the cube.
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Initial point
    x0 = centers.flatten()
    
    # Options
    options = {
        'maxiter': 500,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'disp': False
    }
    
    best_c = centers
    best_score = 0.0
    
    def compute_true_score(c):
        # Compute the actual radius
        # min pairwise dist / 2
        # min boundary dist
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(dist_sq)) / 2.0
        
        abs_c = np.abs(c)
        b_dist = np.min(0.5 - abs_c, axis=1)
        min_b_dist = np.min(b_dist)
        
        return min(min_pair_dist, min_b_dist)
    
    best_score = compute_true_score(centers)
    
    # Run optimization
    # We might need to run this multiple times or with different betas if time permits.
    # For now, one run.
    
    try:
        res = minimize(
            objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            bounds=bounds,
            options=options
        )
        
        if res.success or res.status in [1, 2, 4, 5]: # 1: maxiter, 2: precision, 4: rounding, 5: user abort
            c_opt = res.x.reshape(N, 3)
            # Clip to be safe
            c_opt = np.clip(c_opt, -0.5, 0.5)
            
            score_opt = compute_true_score(c_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_c = c_opt
    except Exception:
        pass
    
    # If time remains, try a random restart
    if time.time() - START_TIME < TIME_LIMIT - 10:
        # Random restart
        # Initialize randomly inside cube
        # Ensure min distance > 0.05 to avoid immediate overlap
        # Simple random placement
        c_rand = np.random.default_rng(123).uniform(-0.4, 0.4, size=(N, 3))
        
        # Check validity
        # If invalid, skip or repair.
        # For simplicity, just run optimizer from here.
        
        x0_rand = c_rand.flatten()
        
        try:
            res2 = minimize(
                objective_and_grad, 
                x0_rand, 
                method='L-BFGS-B', 
                jac=True, 
                bounds=bounds,
                options=options
            )
            
            if res2.success or res2.status in [1, 2, 4, 5]:
                c_opt2 = res2.x.reshape(N, 3)
                c_opt2 = np.clip(c_opt2, -0.5, 0.5)
                
                score_opt2 = compute_true_score(c_opt2)
                if score_opt2 > best_score:
                    best_score = score_opt2
                    best_c = c_opt2
        except Exception:
            pass
            
    return best_c

if __name__ == "__main__":
    construct()
