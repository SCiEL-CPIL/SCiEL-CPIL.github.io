# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize
import random
import sys

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin, evaluator kills at 630s

    n_points = 16

    def compute_score(points):
        if points.shape != (n_points, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def normalize_points(points):
        """Scale points such that the maximum pairwise distance is 1."""
        d = pdist(points)
        d_max = np.max(d)
        if d_max > 1e-12:
            return points / d_max
        return points

    def soft_min_objective_and_grad(x, beta):
        """
        Computes the soft-min objective and its gradient.
        Objective: -soft_min(distances) = (1/beta) * log(sum(exp(-beta * d)))
        We minimize this, so we return (value, gradient).
        """
        points = x.reshape(n_points, 2)
        d = pdist(points)
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0, np.zeros_like(x)
        
        # Normalize distances to [0, 1] to make beta scale invariant
        d_norm = d / d_max
        
        # Calculate weights for soft min
        # To avoid overflow, we can shift, but exp(-beta * d) is safe for d>=0
        exp_vals = np.exp(-beta * d_norm)
        sum_exp = np.sum(exp_vals)
        
        if sum_exp == 0:
            return 0.0, np.zeros_like(x)
            
        # Value: -1/beta * log(sum_exp)
        val = -1.0 / beta * np.log(sum_exp)
        
        # Gradient
        # d_val / d_x_k = (1 / (beta * sum_exp)) * sum_{i<j} d(exp(-beta * d_ij) / d_x_k)
        # = (1 / (beta * sum_exp)) * sum_{i<j} exp(-beta * d_ij) * (-beta) * (d_ij / d_x_k)
        # = -(1 / sum_exp) * sum_{i<j} exp(-beta * d_ij) * (d_ij / d_x_k)
        
        # We need to accumulate gradients for each point coordinate.
        # d_ij / d_x_k is 0 unless k is i or j.
        # If k == i, d_ij / d_x_i = (x_i - x_j) / d_ij
        # If k == j, d_ij / d_x_j = (x_j - x_i) / d_ij
        
        grad = np.zeros_like(x)
        
        # Efficiently compute using matrix operations or loops?
        # N=16, so N^2=256 pairs. Loop is fine.
        
        # We need the indices of the pdist distances
        # pdist returns distances in order: (0,1), (0,2), ..., (0,15), (1,2), ...
        
        idx = 0
        for i in range(n_points):
            for j in range(i + 1, n_points):
                d_ij = d[idx]
                if d_ij > 1e-12:
                    weight = exp_vals[idx] / sum_exp
                    # Gradient contribution to point i
                    # d_val / d_x_i = - weight * (x_i - x_j) / d_ij
                    # Note: d_norm = d / d_max, so d(d_norm)/d(x) = d(d)/d(x) / d_max
                    # The chain rule:
                    # val = -1/beta log(sum exp(-beta d/dmax))
                    # d val / d x = 1/beta * 1/sum * sum exp * (-beta/dmax) * d(d)/dx
                    # = -1/(sum * dmax) * sum exp * d(d)/dx
                    
                    # Let's re-derive carefully.
                    # f = -1/beta log(S), S = sum exp(-beta d_k / D) where D = d_max
                    # df/dx = -1/beta * 1/S * dS/dx
                    # dS/dx = sum exp(-beta d_k / D) * d(-beta d_k / D)/dx
                    # = sum exp(-beta d_k / D) * (-beta/D) * d(d_k)/dx
                    # df/dx = -1/beta * 1/S * (-beta/D) * sum exp * d(d_k)/dx
                    # = 1/(S * D) * sum exp * d(d_k)/dx
                    
                    # So grad = (1 / (sum_exp * d_max)) * sum_{k} exp(-beta d_k/d_max) * d(d_k)/dx
                    
                    vec_ij = points[i] - points[j]
                    grad[i] += weight * (vec_ij / d_ij) / d_max
                    grad[j] += weight * (-vec_ij / d_ij) / d_max
                idx += 1
        
        return val, grad

    def optimize_candidate_gradient(points, time_budget=5.0, betas=None):
        """
        Optimizes a candidate using L-BFGS-B with analytic gradient and increasing beta.
        """
        if betas is None:
            betas = [10, 50, 100, 200, 500, 1000]
            
        points = points.copy()
        points = normalize_points(points)
        
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        start_opt = time.time()
        
        for beta in betas:
            if time.time() - start_opt > time_budget:
                break
            
            def obj_and_grad(x):
                val, grad = soft_min_objective_and_grad(x, beta)
                return val, grad
            
            try:
                res = minimize(
                    obj_and_grad, 
                    points.flatten(), 
                    method='L-BFGS-B', 
                    jac=True, # Use analytic gradient
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
                )
                
                opt_points = res.x.reshape(n_points, 2)
                opt_points = normalize_points(opt_points)
                opt_score = compute_score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    
                points = best_points.copy()
                
            except Exception:
                pass
                
            if time.time() - start_opt > time_budget:
                break
                
        return best_points, best_score

    best_points = None
    best_score = -np.inf
    all_candidates = []

    # 1. Generate Diverse Initial Candidates
    
    # 1a. Square Grid (4x4) - Known strong baseline
    grid_coords = []
    for i in range(4):
        for j in range(4):
            grid_coords.append([i - 1.5, j - 1.5])
    all_candidates.append(np.array(grid_coords))

    # 1b. Triangular Lattice
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice_points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * s + (j % 2) * s / 2
            y = j * h
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    centroid = np.mean(lattice_points, axis=0)
    dists_to_centroid = np.sum((lattice_points - centroid) ** 2, axis=1)
    idx = np.argsort(dists_to_centroid)[:n_points]
    all_candidates.append(lattice_points[idx])

    # 1c. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    poly_points = np.column_stack([np.cos(angles), np.sin(angles)])
    all_candidates.append(poly_points)

    # 1d. Concentric Circles (4 inner, 12 outer)
    inner_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi/12
    inner_radius = 0.4
    outer_radius = 1.0
    inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    all_candidates.append(np.vstack([inner_points, outer_points]))

    # 1e. Random Points
    for seed in range(20):
        np.random.seed(seed * 100 + 7)
        random_points = np.random.randn(n_points, 2)
        random_points = normalize_points(random_points)
        all_candidates.append(random_points)

    # 2. Evaluate and Optimize Candidates
    
    scored_candidates = []
    for cand in all_candidates:
        s = compute_score(cand)
        scored_candidates.append((s, cand))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Optimize top candidates with more time
    for i, (init_score, cand) in enumerate(scored_candidates):
        if time.time() > deadline - 60:
            break
        
        if i < 5:
            budget = 15.0
        elif i < 10:
            budget = 5.0
        else:
            budget = 2.0
            
        opt_cand, opt_score = optimize_candidate_gradient(cand, time_budget=budget)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()

    # 3. Final Polish: Nelder-Mead on the very best point
    if best_points is not None and time.time() < deadline - 10:
        def nm_objective(x):
            points = x.reshape(n_points, 2)
            points = normalize_points(points)
            d = pdist(points)
            d_min = np.min(d)
            d_max = np.max(d)
            if d_max == 0:
                return 0.0
            return -(d_min / d_max) ** 2
        
        bp = normalize_points(best_points).flatten()
        
        try:
            res = minimize(
                nm_objective, 
                bp, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            nm_points = res.x.reshape(n_points, 2)
            nm_points = normalize_points(nm_points)
            nm_score = compute_score(nm_points)
            
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points.copy()
                
        except Exception:
            pass

    # Ensure valid output
    if best_points is None:
        best_points = all_candidates[0]
    
    return best_points

# EVOLVE-BLOCK-END
