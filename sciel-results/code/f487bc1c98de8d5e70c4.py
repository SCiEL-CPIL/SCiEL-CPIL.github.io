"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 120.0  # Safety margin for 150s limit
    
    # 1. Initialization: Hexagonal lattice
    # Estimate spacing for 300 points in unit circle
    # Area = pi. N points. Area per point ~ pi/N. Spacing ~ sqrt(pi/N)
    # Hex lattice density is sqrt(3)/2 * s^2 per point? No, area per point is (sqrt(3)/2) * s^2 is for triangular.
    # Let's just use a fine grid.
    s = 0.05
    points = []
    # Generate hex grid
    for i in range(-int(2/s), int(2/s)+1):
        for j in range(-int(2/s), int(2/s)+1):
            x = i * s + (j % 2) * s * 0.5
            y = j * s * np.sqrt(3) / 2
            if x*x + y*y <= 0.95**2: # Keep well inside initially
                points.append([x, y])
    
    points = np.array(points)
    
    # Sort by radius and take N
    radii = np.linalg.norm(points, axis=1)
    points = points[np.argsort(radii)]
    
    if len(points) < N:
        # Fallback: if lattice is too sparse, use random points
        # But hex lattice with s=0.05 should have plenty.
        # pi * 1^2 / (0.05 * 0.05 * sqrt(3)/2) ~ 3.14 / 0.0043 ~ 700 points.
        pass
    
    x0 = points[:N].copy()
    
    # 2. Optimization
    # Objective: Maximize min distance.
    # Minimize: -soft_min(distances) + penalty
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 2)
        
        # Calculate pairwise distances
        # Use broadcasting for efficiency, but N=300 means 90000 pairs.
        # This is manageable.
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Mask diagonal
        np.fill_diagonal(dist_sq, np.inf)
        
        # Pairwise distances
        dists = np.sqrt(dist_sq)
        # We need min over i<j. The matrix is symmetric.
        # Take upper triangle
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        
        # Boundary distances
        radii = np.linalg.norm(x, axis=1)
        boundary_dists = 1.0 - radii
        
        # Combine all distances
        # We want to maximize the minimum of (pair_dists / 2) and boundary_dists
        # Let's work with squared distances to avoid sqrt in gradient?
        # No, soft-min works fine with distances.
        
        # Penalty for going outside
        outside_penalty = np.sum(np.maximum(0, radii - 1.0)**2)
        
        # Soft min parameters
        # We will pass beta via closure or global? 
        # Better to use a wrapper or default args in minimize.
        # For now, let's assume beta is set in the outer loop and we define the function inside the loop.
        
        return x, pair_dists, boundary_dists, outside_penalty, radii

    # We need to define the objective function that takes beta
    def make_objective(beta):
        def obj(x_flat):
            x = x_flat.reshape(-1, 2)
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            dists = np.sqrt(dist_sq)
            iu = np.triu_indices(N, k=1)
            pair_dists = dists[iu] / 2.0  # Radius of small circle
            
            radii = np.linalg.norm(x, axis=1)
            boundary_dists = 1.0 - radii
            
            # Penalty
            penalty = 1000.0 * np.sum(np.maximum(0, radii - 1.0)**2)
            
            # Soft min
            # Values: pair_dists and boundary_dists
            # We want min(v). Approx: -1/beta * log(sum(exp(-beta * v)))
            # To avoid overflow/underflow, use log-sum-exp trick
            all_dists = np.concatenate([pair_dists, boundary_dists])
            
            # Clip distances to be positive for log? 
            # If a point is outside, boundary_dist < 0.
            # exp(-beta * negative) = exp(positive) -> large.
            # This is correct: it penalizes small/negative distances heavily.
            
            # However, if dist is very large, exp(-beta*dist) -> 0.
            # If dist is very small, exp(-beta*dist) -> 1.
            
            # Numerical stability
            max_val = np.max(all_dists)
            exp_term = np.exp(-beta * (all_dists - max_val))
            sum_exp = np.sum(exp_term)
            if sum_exp == 0:
                soft_min = max_val
            else:
                soft_min = max_val - np.log(sum_exp) / beta
            
            # Minimize -soft_min + penalty
            return -soft_min + penalty
        
        def grad(x_flat):
            x = x_flat.reshape(-1, 2)
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            dists = np.sqrt(dist_sq)
            iu = np.triu_indices(N, k=1)
            pair_dists = dists[iu] / 2.0
            
            radii = np.linalg.norm(x, axis=1)
            boundary_dists = 1.0 - radii
            
            all_dists = np.concatenate([pair_dists, boundary_dists])
            
            # Gradient of soft min
            # d/dx (-1/beta log(sum exp(-beta v))) = 1/beta * sum(exp(-beta v_i) * (-beta) * dv_i/dx) / sum(exp(-beta v_i))
            # = - sum(exp(-beta v_i) * dv_i/dx) / sum(exp(-beta v_i))
            
            max_val = np.max(all_dists)
            exp_term = np.exp(-beta * (all_dists - max_val))
            sum_exp = np.sum(exp_term)
            
            if sum_exp == 0:
                weights = np.zeros_like(all_dists)
                # If sum is 0, it means all exp terms underflowed? 
                # Actually if beta is large and dists are large, exp terms are 0.
                # This shouldn't happen if we have valid distances.
                # If it does, gradient is 0?
                pass
            else:
                weights = exp_term / sum_exp
            
            # Gradient w.r.t x
            g = np.zeros_like(x)
            
            # Pairwise part
            # weights[0:N_pairs] correspond to pair_dists
            # pair_dists = ||x_i - x_j|| / 2
            # d(pair_dist)/dx_i = (x_i - x_j) / (2 * ||x_i - x_j||)
            # Note: pair_dists array is flattened from upper triangle.
            # We need to map back to indices.
            
            # This is inefficient to loop. Let's use vectorized ops.
            # Reconstruct full weight matrix for pairs
            w_pairs = np.zeros((N, N))
            w_pairs[iu] = weights[:len(iu[0])]
            w_pairs = w_pairs + w_pairs.T
            
            # d(pair_dist_ij)/dx_i = (x_i - x_j) / (2 * ||x_i - x_j||)
            # dists is the full matrix of distances (not divided by 2)
            # So d(pair_dist)/dx = (x_i - x_j) / dists[i,j]
            # Wait, pair_dist = dists/2. So d(pair_dist)/dx = 0.5 * (x_i - x_j)/dists[i,j]
            
            # Avoid division by zero
            mask = dists > 1e-8
            safe_dists = np.where(mask, dists, 1.0)
            
            # grad_pair_ij = w_pairs[i,j] * 0.5 * (x_i - x_j) / safe_dists[i,j]
            # Sum over j for each i
            # g_i = sum_j grad_pair_ij
            
            # Vectorized:
            # diff is x_i - x_j
            # We want sum_j w_pairs[i,j] * 0.5 * diff[i,j] / safe_dists[i,j]
            # Let's compute term = w_pairs * 0.5 / safe_dists
            term = w_pairs * 0.5 / safe_dists
            term[~mask] = 0.0
            
            # g_pair = sum_j term[i,j] * (x_i - x_j)
            # This is (term * diff).sum(axis=1) ? No.
            # diff[i,j] is a vector.
            # g_pair[i] = sum_j term[i,j] * diff[i,j]
            
            # Using einsum or broadcasting
            # term shape (N, N), diff shape (N, N, 2)
            # g_pair = np.einsum('ij,ijc->ic', term, diff)
            
            g_pair = np.einsum('ij,ijc->ic', term, diff)
            
            # Boundary part
            # weights[len(iu[0]):] correspond to boundary_dists
            w_bound = weights[len(iu[0]):]
            # boundary_dist = 1 - ||x_i||
            # d(boundary_dist)/dx_i = -x_i / ||x_i||
            # g_bound_i = w_bound[i] * (-x_i / radii[i])
            
            safe_radii = np.where(radii > 1e-8, radii, 1.0)
            g_bound = -w_bound[:, np.newaxis] * (x / safe_radii[:, np.newaxis])
            
            # Penalty gradient
            # penalty = 1000 * sum(max(0, r-1)^2)
            # d/dx = 1000 * 2 * (r-1) * x/r for r>1
            outside_mask = radii > 1.0
            g_pen = np.zeros_like(x)
            if np.any(outside_mask):
                g_pen[outside_mask] = 2000.0 * (radii[outside_mask] - 1.0)[:, np.newaxis] * (x[outside_mask] / safe_radii[outside_mask, np.newaxis])
            
            total_grad = -g_pair - g_bound + g_pen
            # Wait, objective is -soft_min + penalty.
            # grad(-soft_min) = - grad(soft_min) = - ( - sum(...) ) = sum(...) ?
            # Let's re-verify.
            # f = - soft_min.
            # soft_min = -1/beta log(sum exp(-beta v)).
            # grad(soft_min) = -1/beta * 1/sum * sum(exp(-beta v_i) * (-beta) grad(v_i))
            # = 1/sum * sum(exp(-beta v_i) grad(v_i))
            # So grad(-soft_min) = - 1/sum * sum(exp(-beta v_i) grad(v_i))
            # My weights were exp/sum.
            # So grad(-soft_min) = - sum(weights_i * grad(v_i))
            # My calculation: g_pair + g_bound was sum(weights * grad(v)).
            # So I should return - (g_pair + g_bound) + g_pen.
            
            return (-g_pair - g_bound + g_pen).flatten()
        
        return obj, grad

    # Annealing schedule
    betas = [10, 50, 100, 500, 1000, 5000]
    current_x = x0
    
    for beta in betas:
        if time.time() - start_time > time_limit - 10:
            break
            
        obj, grad = make_objective(beta)
        
        # Run L-BFGS-B
        # maxiter depends on time
        # Estimate time per iteration. 
        # N=300, pairwise is 45k. Should be fast.
        try:
            res = minimize(
                obj, 
                current_x.flatten(), 
                jac=grad, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success:
                current_x = res.x.reshape(-1, 2)
            else:
                # If it failed, keep previous
                pass
        except Exception:
            pass

    # Final polish with very high beta
    if time.time() - start_time < time_limit - 5:
        beta_final = 10000
        obj, grad = make_objective(beta_final)
        try:
            res = minimize(
                obj, 
                current_x.flatten(), 
                jac=grad, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.success:
                current_x = res.x.reshape(-1, 2)
        except Exception:
            pass

    # Ensure validity: clip to circle?
    # The penalty should have kept them inside.
    # But just in case, project back if slightly outside.
    radii = np.linalg.norm(current_x, axis=1)
    outside = radii > 1.0
    if np.any(outside):
        current_x[outside] /= radii[outside, np.newaxis]
        # This might cause overlaps, but it's better than invalid.
        # Actually, if they are outside, the score is 0.
        # The penalty should prevent this.
        
    return current_x

# EVOLVE-BLOCK-END
