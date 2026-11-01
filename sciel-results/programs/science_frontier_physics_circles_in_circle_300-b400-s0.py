"""Improved program for packing 300 circles in a unit circle using smooth soft-min optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    points = []
    # Estimate spacing. r ~ 1/sqrt(N). Diameter ~ 2r.
    # Hex lattice spacing s = 2r.
    # Area of hexagon per point ~ s^2 * sqrt(3)/2.
    # N * s^2 * sqrt(3)/2 ~ pi.
    # s ~ sqrt(2*pi / (N*sqrt(3))).
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate a large grid of hex lattice points
    # Hex lattice coordinates:
    # x = s * (i + 0.5 * (j % 2))
    # y = s * j * sqrt(3) / 2
    max_k = int(1.0 / (s * 0.5)) + 2
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        # If we didn't generate enough, shrink spacing slightly and retry
        # This is a fallback, the estimate should be close
        s *= 0.9
        max_k = int(1.0 / (s * 0.5)) + 3
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        # Sort by distance to origin to pick the most central ones?
        # Or just pick the first n? Picking central ones might be better for stability.
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        # Fallback: random points
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    # Add small random perturbation to break symmetry
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # 2. Optimization using L-BFGS-B with smooth soft-min objective
    # Objective: minimize -soft_min(dists, boundary_dists)
    # soft_min(z) = -1/beta * log(sum(exp(-beta * z)))
    # Gradient: grad(soft_min) = sum(exp(-beta*z) * grad(z)) / sum(exp(-beta*z))
    #           = sum(p_k * grad(z_k)) where p_k = exp(-beta*z_k) / sum(exp(-beta*z))
    
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Calculate distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # We want the radius r, so the "distance" for pairs is d/2.
        # Let's define the "constraint distances" c_k.
        # For pairs (i,j): c_{ij} = dists[i,j] / 2.0
        # For boundary i: c_i = 1.0 - norms[i]
        
        # We want to maximize min(c_k).
        # Minimize -min(c_k).
        # Smooth approximation: -soft_min(c)
        
        # Flatten distances for vectorization
        # Pair distances: upper triangle
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu] / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # Combine all constraint distances
        # Note: pair_dists has N*(N-1)/2 elements, bound_dists has N elements.
        # We need to compute soft_min and its gradient.
        
        # To avoid overflow in exp, we use the log-sum-exp trick.
        # Let z = -beta * c.
        # soft_min(c) = -1/beta * log(sum(exp(-beta*c)))
        
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        # Concatenate for sum
        # We can compute log_sum_exp separately or combined.
        # Combined is easier for gradient weighting.
        
        # Max values for stability
        max_z_pair = np.max(z_pairs)
        max_z_bound = np.max(z_bounds)
        max_z = max(max_z_pair, max_z_bound)
        
        # exp terms
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        # log_sum_exp = max_z + log(sum_exp)
        log_sum_exp = max_z + np.log(sum_exp)
        
        # Objective value: -soft_min = 1/beta * log_sum_exp
        val = log_sum_exp / beta
        
        # Gradient
        # grad(soft_min) = (1/sum_exp) * sum_k exp(-beta*c_k) * grad(c_k)
        # grad(-soft_min) = -grad(soft_min)
        # We want grad of objective = grad(-soft_min)
        
        # Weights
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        # Pair contributions
        # c_{ij} = 0.5 * |p_i - p_j|
        # grad_{p_i} c_{ij} = 0.5 * (p_i - p_j) / |p_i - p_j|
        # grad_{p_j} c_{ij} = 0.5 * (p_j - p_i) / |p_i - p_j|
        
        # Vectorized pair gradient
        # For each pair (i,j) with weight w_k:
        # grad[p_i] += w_k * 0.5 * (p_i - p_j) / dists[i,j]
        # grad[p_j] += w_k * 0.5 * (p_j - p_i) / dists[i,j]
        
        # dists[i,j] = 2 * pair_dists[k]
        # So 0.5 / dists[i,j] = 0.5 / (2 * pair_dists[k]) = 0.25 / pair_dists[k]
        
        # Let's compute the vector directions
        # diff[i,j] = p_i - p_j
        # We have iu indices.
        i_idx, j_idx = iu
        
        # Avoid division by zero
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        
        # Coefficient for each pair
        # w_k * 0.25 / pair_dists[k]
        coeff = w_pairs * 0.25 / safe_pair_dists
        
        # Vector from j to i
        vecs = p[i_idx] - p[j_idx]
        
        # Accumulate gradients
        # grad[p_i] += coeff * vecs
        # grad[p_j] -= coeff * vecs
        
        np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
        
        # Boundary contributions
        # c_i = 1 - |p_i|
        # grad_{p_i} c_i = - p_i / |p_i|
        
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = -p / safe_norms[:, np.newaxis]
        
        # grad[p_i] += w_bounds[i] * bound_grad_vecs[i]
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        # The objective is -soft_min, so gradient is -grad(soft_min)
        # Wait, my derivation:
        # f = -soft_min
        # grad(f) = -grad(soft_min)
        # grad(soft_min) = sum(w_k * grad(c_k))
        # So grad(f) = -sum(w_k * grad(c_k))
        
        return val, -grad

    # Optimization loop with increasing beta
    # Start with low beta (smooth), increase to high beta (sharp)
    betas = [10.0, 50.0, 100.0, 500.0, 1000.0]
    
    current_points = points.copy()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'gtol': 1e-8,
                    'ftol': 1e-10,
                    'maxls': 20,
                    'disp': False
                }
            )
            
            if time.time() > deadline - 5:
                break
                
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
                current_points = candidate
            else:
                # If it didn't improve, keep current best for next beta
                # But maybe the optimizer moved it to a worse local min.
                # We should probably restart from best_points for next beta?
                # Or just continue from current_points if it's close.
                # Let's stick with best_points for stability.
                current_points = best_points
                
        except Exception:
            pass

    # Final polish with very high beta if time permits
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: objective_and_grad(x, 5000.0)[0],
                best_points.flatten(),
                jac=lambda x: objective_and_grad(x, 5000.0)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 100,
                    'gtol': 1e-10,
                    'ftol': 1e-12,
                    'maxls': 10,
                    'disp': False
                }
            )
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
        except Exception:
            pass

    # Ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    max_allowed = 1.0 - best_score
    # If any point is slightly outside due to numerical error, clip it
    # But clipping might cause overlap. 
    # The soft-min objective should keep them well inside.
    # Just to be safe, if any norm > 1, scale back.
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
        # Recalculate score
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
