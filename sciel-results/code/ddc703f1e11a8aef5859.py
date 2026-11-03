import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    t_start = time.time()
    time_limit = 140.0  # Leave 10s margin
    
    # 1. Initial Guess: Hexagonal Lattice
    # We want to pack 50 points.
    # A hex grid with spacing s.
    # Number of points in a circle of radius R is approx pi * R^2 / (s^2 * sqrt(3)/2)
    # Here R=1. N=50.
    # 50 = pi / (s^2 * 0.866) => s^2 = pi / (50 * 0.866) = 0.0725 => s = 0.269
    # Let's start with s = 0.26
    
    s = 0.26
    # Generate grid points
    points = []
    # Range needs to cover radius 1
    # Max index k such that k*s < 1
    k_max = int(1.0 / s) + 2
    
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2.0
            points.append((x, y))
            
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    
    # Sort by radius and take closest N points
    # This ensures we have a "disk" of points
    indices = np.argsort(radii)
    init_points = points[indices[:N]]
    
    # Flatten for optimizer
    x0 = init_points.flatten()
    
    # 2. Objective Function
    # We want to maximize r = min(min_pairwise_dist/2, min_boundary_dist)
    # Let's minimize -r_smooth
    
    beta = 50.0  # Sharpness of soft-min
    
    def objective_and_grad(x):
        p = x.reshape(N, 2)
        
        # Pairwise distances
        # dist[i, j] = |p_i - p_j|
        # We only need i < j
        # Using broadcasting
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        
        # Extract upper triangle
        iu = np.triu_indices(N, k=1)
        pairwise_dists = dist[iu]
        half_pairwise = pairwise_dists / 2.0
        
        # Boundary distances
        # dist to boundary = 1 - |p_i|
        radii_p = np.sqrt(np.sum(p**2, axis=1))
        boundary_dists = 1.0 - radii_p
        
        # Combine all constraints into a single vector D
        D = np.concatenate([half_pairwise, boundary_dists])
        
        # Soft min
        # r = -1/beta * log(sum(exp(-beta * D)))
        # To avoid overflow, subtract max D? No, we want min D.
        # exp(-beta * D) is largest for smallest D.
        # Let's use log-sum-exp stability.
        # max_D = np.max(D)
        # term = -beta * (D - max_D)  -> This shifts D so max D becomes 0, but we want min D to be large?
        # Actually, standard log-sum-exp:
        # sum(exp(-beta D)) = exp(-beta * D_max) * sum(exp(-beta (D - D_max)))
        # log(sum(...)) = -beta D_max + log(sum(exp(-beta (D - D_max))))
        # r = D_max - 1/beta * log(sum(exp(-beta (D - D_max))))
        
        D_max = np.max(D)
        shifted = -beta * (D - D_max)
        # shifted will have 0 for the max D, and negative for others.
        # Wait, if D is small, -beta*(D-D_max) is large positive.
        # We want to sum exp(-beta D).
        # Let's just compute exp(-beta * D) directly if D is not too large.
        # D is roughly 0.1 to 1.0. beta=50. -beta*D is -5 to -50.
        # exp(-50) is very small. exp(-5) is 0.006.
        # It is numerically stable enough without shifting if D > 0.
        # However, to be safe:
        
        exp_terms = np.exp(-beta * D)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
             return 0, np.zeros_like(x)
             
        r_smooth = -1.0 / beta * np.log(sum_exp)
        
        # Gradient
        # dr/dx = 1/beta * 1/sum_exp * sum( exp(-beta D_k) * (-beta * dD_k/dx) ) * (-1) ?
        # Let's derive carefully.
        # f = -r_smooth = 1/beta * log(sum_exp)
        # df/dx = 1/beta * 1/sum_exp * d(sum_exp)/dx
        # d(sum_exp)/dx = sum( d(exp(-beta D_k))/dx )
        # = sum( exp(-beta D_k) * (-beta * dD_k/dx) )
        # df/dx = 1/beta * 1/sum_exp * (-beta) * sum( exp(-beta D_k) * dD_k/dx )
        # = -1/sum_exp * sum( exp(-beta D_k) * dD_k/dx )
        
        # We need dD_k/dx for each constraint k.
        # D consists of half_pairwise and boundary_dists.
        
        grad = np.zeros_like(x)
        
        # Boundary part
        # D_b_i = 1 - |p_i|
        # dD_b_i/dp_i = - p_i / |p_i|
        # Contribution to grad: -1/sum_exp * exp(-beta D_b_i) * (- p_i / |p_i|)
        # = 1/sum_exp * exp(-beta D_b_i) * p_i / |p_i|
        
        # Pairwise part
        # D_p_ij = 0.5 * |p_i - p_j|
        # dD_p_ij/dp_i = 0.5 * (p_i - p_j) / |p_i - p_j|
        # dD_p_ij/dp_j = 0.5 * (p_j - p_i) / |p_i - p_j|
        # Contribution to grad:
        # -1/sum_exp * exp(-beta D_p_ij) * (0.5 * (p_i - p_j) / |p_i - p_j|) for i
        # -1/sum_exp * exp(-beta D_p_ij) * (0.5 * (p_j - p_i) / |p_i - p_j|) for j
        
        # Let's compute weights w_k = exp(-beta D_k) / sum_exp
        w = exp_terms / sum_exp
        
        # Accumulate gradient
        # Boundary
        for i in range(N):
            if radii_p[i] > 1e-9:
                grad[2*i:2*i+2] += w[N + i] * (p[i] / radii_p[i])
            # Note: The formula for df/dx had a minus sign in front of the sum.
            # df/dx = - sum(w_k * dD_k/dx)
            # So we should subtract.
            # Let's re-verify sign.
            # f = -r. We want to minimize f.
            # r increases when D increases.
            # If we move p away from boundary, D_b increases, r increases, f decreases.
            # Gradient of f should point in direction of decreasing f.
            # So if D_b increases, f decreases.
            # dD_b/dp = -p/|p|.
            # df/dp = -w * dD_b/dp = -w * (-p/|p|) = w * p/|p|.
            # So we ADD w * p/|p|.
            # Correct.
            
        # Pairwise
        # We have vector w_p of length M (number of pairs).
        # w_p[k] corresponds to pair (i, j).
        # dD_p/dp_i = 0.5 * (p_i - p_j) / dist_ij
        # dD_p/dp_j = 0.5 * (p_j - p_i) / dist_ij
        # df/dp_i += -w_p[k] * 0.5 * (p_i - p_j) / dist_ij
        # df/dp_j += -w_p[k] * 0.5 * (p_j - p_i) / dist_ij
        
        w_p = w[:N*(N-1)//2]
        
        # To do this efficiently, we can use the full matrix of weights.
        # Let's create a matrix W of size N x N.
        # W[i, j] = w_p[k] if (i, j) is the k-th pair in upper triangle, else 0.
        # Actually, we can just loop or use broadcasting.
        # Given N=50, M=1225. Looping is fine.
        
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                d = dist[i, j]
                if d < 1e-9:
                    continue
                w_val = w_p[idx]
                # Gradient contribution to i
                grad[2*i:2*i+2] -= w_val * 0.5 * (p[i] - p[j]) / d
                # Gradient contribution to j
                grad[2*j:2*j+2] -= w_val * 0.5 * (p[j] - p[i]) / d
                idx += 1
                
        return -r_smooth, grad # Wait, objective is -r_smooth?
        # No, I defined f = -r_smooth in the derivation.
        # So return f and grad_f.
        # f = -r_smooth.
        # So return -r_smooth, grad.

    # Run optimization
    best_x = x0
    best_val = objective_and_grad(x0)[0]
    
    # Try a few methods or just L-BFGS-B
    # L-BFGS-B is good for large N.
    
    for attempt in range(3): # Multiple restarts with slight perturbation
        if time.time() - t_start > time_limit - 5:
            break
            
        x_start = x0.copy()
        if attempt > 0:
            # Add small noise
            x_start += np.random.normal(0, 0.01, x_start.shape)
            
        res = minimize(
            objective_and_grad, 
            x_start, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    # Final check: compute actual score
    p = best_x.reshape(N, 2)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = np.min(dist[iu]) / 2.0
    radii_p = np.sqrt(np.sum(p**2, axis=1))
    min_bound = np.min(1.0 - radii_p)
    
    r = min(min_pair, min_bound)
    
    # If invalid, return initial
    if not np.isfinite(r) or r <= 0:
         return init_points
         
    return best_x.reshape(N, 2)
