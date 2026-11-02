import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Initialization: Hexagonal lattice
    # We need to generate enough points.
    # The density of hex lattice is 1 point per (sqrt(3)/2 * s^2) area?
    # Area of unit circle is pi.
    # We need 300 points.
    # Let's generate a grid and filter.
    
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    points = None
    
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        pts, radius = pts[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(pts) >= N:
            # Sort by distance to center, take closest N
            pts = pts[np.argsort(radius, kind="stable")[:N]]
            points = pts
            break
        spacing *= 0.98
        
    # 2. Optimization using Soft-Min Surrogate
    # Objective: Minimize -softmin(distances)
    # softmin(d) = -1/tau * log(sum(exp(-tau * d)))
    # We want to maximize min(d), so we minimize -softmin(d)
    
    def objective_and_grad(x):
        # x is shape (N, 2)
        # Calculate pairwise distances
        # d[i,j] = ||x_i - x_j||
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal (distance 0)
        np.fill_diagonal(dist, np.inf)
        
        # Boundary distances
        rad = np.linalg.norm(x, axis=1)
        bound = 1.0 - rad
        
        # Combine distances
        # We want softmin of all pairwise and boundary distances.
        # To avoid memory issues with exp(-tau * dist) if dist is large, we can clip.
        # But dist is small.
        
        tau = 100.0  # Temperature for soft-min. Higher is sharper.
        
        # Calculate soft-min term
        # log_sum_exp(-tau * dist)
        # For numerical stability: max_val = max(-tau * dist)
        # sum(exp(-tau * dist - max_val))
        
        all_dists = np.concatenate([dist[dist < np.inf].flatten(), bound])
        
        if len(all_dists) == 0:
             return 0.0, np.zeros_like(x)
             
        # Clip to avoid overflow in exp
        # If tau * d is very large, exp is 0.
        # We only care about the smallest distances.
        
        # Gradient calculation
        # This is complex to vectorize fully for the log-sum-exp derivative.
        # Let's use a simpler approach if time is tight, or stick to a simpler local search.
        
        # Given the time limit and complexity, a simpler local search might be more robust.
        # Let's try a simple repulsion local search.
        pass

    # Let's switch to a simpler, robust local search:
    # Iterate, for each point, try to move it to increase the min distance.
    
    def calc_score(pts):
        # Returns the current radius
        if len(pts) < 2:
            return min(1 - np.linalg.norm(pts, axis=1)) if len(pts) > 0 else 0.0
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_pair, min_bound)
        
    current_pts = points.copy()
    current_score = calc_score(current_pts)
    
    # Local Search
    # Move points towards the center of their Voronoi cell?
    # Or just random jitter + hill climbing?
    
    # A better heuristic: 
    # For each point, calculate the vector sum of forces from neighbors.
    # Force from neighbor j: (x_i - x_j) / ||x_i - x_j||^2 (Repulsion)
    # Force from boundary: -x_i / ||x_i||^2 (Attraction to center if near boundary? No, repulsion from boundary)
    # Actually, we want to maximize min distance.
    # We can move points in the direction of the gradient of the soft-min.
    
    # Let's implement the gradient of the soft-min properly.
    
    def softmin_grad(x, tau=50.0):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        rad = np.linalg.norm(x, axis=1)
        bound = 1.0 - rad
        
        # We want to maximize m = min(dist/2, bound)
        # Let's work with d = min(dist, 2*bound) to unify scale?
        # No, keep them separate.
        
        # Softmin of set S = {d_ij} U {b_i}
        # f(S) = -1/tau * log(sum(exp(-tau * s)))
        # grad f = 1/tau * sum(exp(-tau * s) * grad s)
        
        # Calculate weights
        # To avoid underflow, we can ignore terms with s > 1/tau * 10?
        
        # Pairwise terms
        # d_ij
        # w_ij = exp(-tau * d_ij)
        
        # Boundary terms
        # b_i = 1 - ||x_i||
        # grad b_i = -x_i / ||x_i||
        # w_i = exp(-tau * b_i)
        
        # Summing up
        grad = np.zeros_like(x)
        
        # Pairwise
        # exp_term = np.exp(-tau * dist)
        # This is an N x N matrix.
        # grad_i += sum_j ( exp(-tau * d_ij) * (x_i - x_j) / d_ij )
        
        # Vectorization:
        # D = dist
        # E = exp(-tau * D)
        # E[diag] = 0
        # Factor = E / D  (set D=0 to 1 to avoid div 0, then mask)
        # Grad_i = sum_j Factor[i,j] * (x_i - x_j)
        
        # Let's compute this
        D = dist
        E = np.exp(-tau * D)
        np.fill_diagonal(E, 0.0)
        
        # Avoid division by zero
        D_safe = np.where(D > 1e-8, D, 1.0)
        Factor = E / D_safe
        
        # Grad_pair[i] = sum_j Factor[i,j] * (x_i - x_j)
        # = x_i * sum_j Factor[i,j] - sum_j Factor[i,j] * x_j
        
        sum_factor = np.sum(Factor, axis=1)
        weighted_sum_x = Factor @ x
        
        grad_pair = x * sum_factor[:, np.newaxis] - weighted_sum_x
        
        # Boundary
        # b_i = 1 - ||x_i||
        # E_b = exp(-tau * b_i)
        # Grad_bound_i = E_b[i] * (-x_i / ||x_i||)
        
        E_b = np.exp(-tau * bound)
        rad_safe = np.where(rad > 1e-8, rad, 1.0)
        grad_bound = -E_b[:, np.newaxis] * x / rad_safe[:, np.newaxis]
        
        total_grad = grad_pair + grad_bound
        
        # We want to MAXIMIZE softmin, so we MINIMIZE -softmin.
        # Gradient of -softmin is -total_grad.
        # So we return -total_grad for minimize.
        
        return -total_grad
        
    # Run optimization
    # L-BFGS-B
    # Bounds: x in [-1, 1], y in [-1, 1]
    bounds = [(-1.0, 1.0)] * (N * 2)
    
    x0 = current_pts.flatten()
    
    # Check time
    if time.time() - start_time > deadline - 10:
        return current_pts
        
    try:
        res = minimize(
            lambda x: 0.0, # We only need gradient for direction? No, minimize needs function.
            x0,
            jac=softmin_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-5}
        )
        
        # The function value is not calculated by minimize if we pass 0.0?
        # We should pass a dummy function or calculate it.
        # Actually, minimize uses the function to check convergence.
        # Let's define a dummy function that returns 0, but uses the gradient.
        # This is risky. Better to define the actual objective.
        
        def objective(x):
            # Just return 0, we rely on gradient?
            # No, L-BFGS-B needs function values.
            # Let's calculate the softmin value.
            pts = x.reshape(-1, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            rad = np.linalg.norm(pts, axis=1)
            bound = 1.0 - rad
            all_dists = np.concatenate([dist[dist < np.inf].flatten(), bound])
            # Softmin
            tau = 50.0
            # max for stability
            m = np.max(-tau * all_dists)
            return -(-1.0/tau * np.log(np.sum(np.exp(-tau * all_dists - m))))
            
        # Re-run with proper objective
        res = minimize(
            objective,
            x0,
            jac=softmin_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-5}
        )
        
        if res.success and time.time() - start_time < deadline - 5:
            new_pts = res.x.reshape(-1, 2)
            if calc_score(new_pts) > current_score:
                current_pts = new_pts
                current_score = calc_score(current_pts)
                
    except Exception as e:
        pass
        
    return current_pts
