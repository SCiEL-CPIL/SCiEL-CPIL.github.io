import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_hull_radius(pts):
    """
    Compute the exact covering radius using the convex hull method.
    Returns the radius in radians.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        return np.max(ang_radii)
    except Exception:
        return np.pi

def _soft_covering_radius(pts, grid_pts, temperature):
    """
    Compute a differentiable surrogate for the covering radius.
    Uses a soft-min (or rather, max of soft-mins) formulation.
    d_ij = angular distance between pts[i] and grid_pts[j]
    soft_min_i(d_ij) = -T * log(sum_i exp(-d_ij / T))
    Objective = max_j soft_min_i(d_ij)
    """
    # dots[i, j] = pts[i] @ grid_pts[j]
    dots = pts @ grid_pts.T
    cos_theta = np.clip(dots, -1.0, 1.0)
    # d_ij = arccos(cos_theta)
    d = np.arccos(cos_theta)
    
    # For each grid point j, we want the min over i of d[i,j]
    # soft_min = -T * log( sum_i exp(-d[i,j] / T) )
    # To avoid overflow in exp(-d/T) when T is small and d is large, 
    # we can use the log-sum-exp trick, but d is non-negative.
    # exp(-d/T) is in (0, 1].
    
    # Vectorized computation
    # d is (N, G)
    # We want row-wise log-sum-exp of (-d/T)
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    # Here x = -d/T. max(x) = -min(d)/T.
    
    min_d = np.min(d, axis=0) # (G,)
    # soft_min_j = -T * ( -min_d_j/T + log(sum_i exp((-d_ij + min_d_j)/T)) )
    # soft_min_j = min_d_j - T * log(sum_i exp((min_d_j - d_ij)/T))
    
    # (min_d_j - d_ij) is <= 0. exp is stable.
    shifted_d = (min_d[:, np.newaxis] - d) / temperature  # (N, G)
    # shifted_d is <= 0.
    exp_shifted = np.exp(shifted_d)
    sum_exp = np.sum(exp_shifted, axis=0) # (G,)
    
    # Avoid log(0)
    sum_exp = np.clip(sum_exp, _EPS, None)
    
    soft_mins = min_d - temperature * np.log(sum_exp)
    
    # The objective is the maximum over grid points
    # To make it differentiable, we can use a soft-max as well, or just return the max
    # and let the optimizer handle the non-differentiability? 
    # Better: use a soft-max for the outer max too?
    # Or simply return the max, and use subgradient? 
    # L-BFGS-B requires gradients. 
    # Let's return the max value, but compute gradient via the soft-min chain.
    # Actually, max_j f(j) is not differentiable. 
    # We can approximate max_j f(j) with soft-max_j f(j) = T2 * log(sum_j exp(f(j)/T2)).
    # Let's use a small T2 for the outer max.
    
    T2 = 0.01
    max_soft = T2 * np.log(np.sum(np.exp(soft_mins / T2)))
    
    return max_soft

def _soft_covering_radius_and_grad(pts, grid_pts, temperature):
    """
    Compute the differentiable surrogate and its gradient.
    """
    dots = pts @ grid_pts.T
    cos_theta = np.clip(dots, -1.0, 1.0)
    d = np.arccos(cos_theta)
    
    min_d = np.min(d, axis=0)
    shifted_d = (min_d[:, np.newaxis] - d) / temperature
    exp_shifted = np.exp(shifted_d)
    sum_exp = np.sum(exp_shifted, axis=0)
    sum_exp = np.clip(sum_exp, _EPS, None)
    
    # p[i, j] = exp_shifted[i, j] / sum_exp[j]
    # This is the soft-attention weight of point i for grid point j
    p = exp_shifted / sum_exp[:, np.newaxis]
    
    # soft_min_j = sum_i p[i,j] * d[i,j] - temperature * sum_i p[i,j] * log(sum_exp[j])? 
    # No, the formula was: soft_min_j = min_d_j - T * log(sum_exp_j)
    # Let's re-derive gradient.
    # f_j = -T * log( sum_i exp(-d_ij/T) )
    # df_j / dd_ij = -T * ( 1/sum * (-1/T) * exp(-d_ij/T) ) = exp(-d_ij/T) / sum_exp_j = p[i,j]
    # So grad_d = p.
    
    # d is arccos(dots).
    # dd / ddots = -1 / sqrt(1 - dots^2)
    # Let denom = sqrt(1 - dots^2)
    denom = np.sqrt(np.clip(1.0 - dots * dots, _EPS, None))
    grad_d_dots = -1.0 / denom
    
    # grad_dots = p * grad_d_dots
    grad_dots = p * grad_d_dots
    
    # Now, we need the gradient of the outer max.
    # Let g_j = soft_min_j.
    # Objective L = max_j g_j.
    # Approximate with soft-max: L = T2 * log(sum_j exp(g_j/T2))
    # q_j = exp(g_j/T2) / sum_k exp(g_k/T2)
    
    T2 = 0.01
    exp_g = np.exp(g_j / T2) if False else np.exp(soft_mins / T2) # Need soft_mins
    # Let's recompute soft_mins for clarity
    soft_mins = min_d - temperature * np.log(sum_exp)
    
    exp_g = np.exp(soft_mins / T2)
    sum_exp_g = np.sum(exp_g)
    q = exp_g / sum_exp_g
    
    # grad_L / grad_g_j = q_j
    # grad_g_j / grad_d_ij = p[i,j]
    # So grad_L / grad_d_ij = q_j * p[i,j]
    
    grad_L_d = q[:, np.newaxis] * p
    
    # grad_L_dots = grad_L_d * grad_d_dots
    grad_L_dots = grad_L_d * grad_d_dots
    
    # grad_L_pts: 
    # dots = pts @ grid_pts.T
    # d(dots)/d(pts[i]) = grid_pts[:, j]
    # grad_L_pts[i] = sum_j grad_L_dots[i,j] * grid_pts[:,j]
    # grad_L_pts = grad_L_dots @ grid_pts
    
    grad_pts = grad_L_dots @ grid_pts
    
    # Project gradient to tangent space of sphere
    # grad_tangent = grad_pts - (grad_pts . pts) * pts
    dot_grad_pts = np.sum(grad_pts * pts, axis=1, keepdims=True)
    grad_tangent = grad_pts - dot_grad_pts * pts
    
    value = T2 * np.log(sum_exp_g)
    
    return value, grad_tangent

def _polish_with_hull(pts, deadline):
    """
    Perform high-iteration count of facet-based local optimization.
    Moves the worst facet's vertices towards its pole.
    """
    best_pts = pts.copy()
    best_r = _compute_hull_radius(pts)
    
    step = 0.02
    no_improve_count = 0
    
    while time.time() < deadline:
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            M = len(faces)
            
            A = pts[faces[:, 0]]
            B = pts[faces[:, 1]]
            C = pts[faces[:, 2]]
            
            AB = B - A
            AC = C - A
            cross = np.cross(AB, AC)
            area = 0.5 * np.linalg.norm(cross, axis=1)
            valid = area > _EPS
            if not np.any(valid):
                break
                
            normals = cross[valid]
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            centroids = (A[valid] + B[valid] + C[valid]) / 3.0
            signs = np.sum(normals * centroids, axis=1, keepdims=True)
            normals = normals * np.sign(signs)
            normals = normals / (norms + _EPS)
            
            cos_theta = np.sum(normals * A[valid], axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            ang_radii = np.arccos(cos_theta)
            
            curr_r = np.max(ang_radii)
            
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 20:
                    step *= 0.95
                    no_improve_count = 0
                    if step < 1e-5:
                        break
            
            worst_idx = np.argmax(ang_radii)
            v_indices = faces[valid][worst_idx]
            pole = normals[worst_idx]
            
            cand = pts.copy()
            for v in v_indices:
                dir_vec = pole - pts[v]
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > _EPS:
                    dir_vec /= dir_norm
                else:
                    dir_vec = 0.0
                cand[v] = pts[v] + step * dir_vec
                cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
                
            cand_r = _compute_hull_radius(cand)
            if cand_r < curr_r:
                pts = cand
            else:
                step *= 0.8
                
        except Exception:
            break
            
    return best_pts, best_r

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    # Create a dense spherical grid
    # 150x300 grid suggested in knowledge
    G_theta = 150
    G_phi = 300
    theta = np.linspace(0, np.pi, G_theta, endpoint=False) + np.pi / G_theta
    phi = np.linspace(0, 2 * np.pi, G_phi, endpoint=False)
    
    th, ph = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(th) * np.cos(ph)
    y = np.sin(th) * np.sin(ph)
    z = np.cos(th)
    
    grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid_pts /= np.linalg.norm(grid_pts, axis=1, keepdims=True)
    
    best_pts = _initial_fib()
    best_r = _compute_hull_radius(best_pts)
    
    # Multi-restart with Gradient Descent
    n_restarts = 8
    time_per_restart = (deadline - 10.0) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            pts = _initial_fib()
        else:
            # Add noise to best found so far or random?
            # Random might be better for diversity, but best+noise might be faster to converge
            # Let's try random for first few, then best+noise
            if restart < 3:
                pts = _random_init(rng)
            else:
                pts = best_pts + rng.standard_normal((N, 3)) * 0.1
                pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        curr_r = _compute_hull_radius(pts)
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
        
        # Gradient Descent on Soft Covering
        # Parameters
        T_start = 0.1
        T_end = 0.01
        lr = 0.01
        max_iters = 500
        
        # We will run L-BFGS-B or manual GD.
        # Manual GD is easier to control for time.
        
        curr_T = T_start
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            val, grad = _soft_covering_radius_and_grad(pts, grid_pts, curr_T)
            
            # Move in negative gradient direction
            # Project to tangent space
            # grad is already projected
            
            # Adaptive step size?
            # Simple line search or fixed step
            step = lr * (curr_T / T_start) + 0.001
            
            cand = pts - step * grad
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            
            # Check improvement on exact metric? 
            # Exact metric is expensive (ConvexHull).
            # Use soft metric for acceptance, but track exact best.
            cand_val, _ = _soft_covering_radius_and_grad(cand, grid_pts, curr_T)
            
            # Accept if soft value improved
            if cand_val < val:
                pts = cand
                # Occasionally check exact radius
                if it % 10 == 0:
                    exact_r = _compute_hull_radius(pts)
                    if exact_r < best_r:
                        best_r = exact_r
                        best_pts = pts.copy()
            else:
                # Reduce step if stuck
                lr *= 0.9
            
            # Cool down temperature
            curr_T *= 0.98
            if curr_T < T_end:
                break
    
    # Final local polish using exact hull metric
    if time.time() < deadline - 2.0:
        best_pts, best_r = _polish_with_hull(best_pts, deadline)
        
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
