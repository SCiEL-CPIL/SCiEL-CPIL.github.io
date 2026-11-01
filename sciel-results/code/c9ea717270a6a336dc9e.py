"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    best_score = -1.0
    best_points = None
    
    def calculate_score(points):
        if points is None:
            return 0.0
        pts = points.reshape(N, 3)
        if np.any(pts < -0.5 - 1e-9) or np.any(pts > 0.5 + 1e-9):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
            
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        r = min(min_pair / 2.0, min_boundary)
        return r

    def objective_and_grad(x, P):
        pts = x.reshape(N, 3)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Soft minimum approximation using log-sum-exp
        dists_safe = np.maximum(dists, 1e-10)
        
        # Objective: minimize sum of (1/d)^P
        # This is equivalent to minimizing -log(sum(exp(-P * log(d))))
        # But for numerical stability, we use the direct power form
        sum_pair = np.sum(dists_safe ** (-P))
        
        # Gradient of pairwise part
        # d/dx_i sum_j (1/d_ij^P) = -P * sum_j (1/d_ij^(P+2)) * (x_i - x_j)
        W = dists_safe ** (-P - 2.0)
        np.fill_diagonal(W, 0.0)
        
        row_sums = np.sum(W, axis=1)
        grad_pair = -P * (row_sums[:, np.newaxis] * pts - (W @ pts))
        
        # Boundary part
        abs_pts = np.abs(pts)
        b_dists = 0.5 - abs_pts
        min_idx = np.argmin(b_dists, axis=1)
        min_b_vals = np.min(b_dists, axis=1)
        
        min_b_safe = np.maximum(min_b_vals, 1e-10)
        sum_bound = np.sum(min_b_safe ** (-P))
        
        # Gradient of boundary part
        grad_bound = np.zeros_like(pts)
        for i in range(N):
            k = min_idx[i]
            s = np.sign(pts[i, k])
            if s == 0:
                s = 1.0
            # d/dx (1/(0.5-|x|)^P) = P * (0.5-|x|)^(-P-1) * sign(x)
            grad_bound[i, k] = P * (min_b_safe[i] ** (-P - 1.0)) * s
        
        total_grad = grad_pair + grad_bound
        total_obj = sum_pair + sum_bound
        
        return total_obj, total_grad.flatten()

    def init_grid_5x5x4():
        tx = (np.arange(5) + 0.5) / 5 - 0.5
        ty = (np.arange(5) + 0.5) / 5 - 0.5
        tz = (np.arange(4) + 0.5) / 4 - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        return grid.flatten()
    
    def init_grid_4x5x5():
        tx = (np.arange(4) + 0.5) / 4 - 0.5
        ty = (np.arange(5) + 0.5) / 5 - 0.5
        tz = (np.arange(5) + 0.5) / 5 - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        return grid.flatten()
    
    def init_grid_5x4x5():
        tx = (np.arange(5) + 0.5) / 5 - 0.5
        ty = (np.arange(4) + 0.5) / 4 - 0.5
        tz = (np.arange(5) + 0.5) / 5 - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        return grid.flatten()
    
    def init_random():
        return np.random.uniform(-0.45, 0.45, size=(N, 3)).flatten()
    
    def init_fibonacci():
        pts = np.zeros((N, 3))
        golden = (1 + 5**0.5) / 2
        for i in range(N):
            t = i / N
            z = 2 * t - 1
            r = np.sqrt(max(0, 1 - z*z))
            theta = 2 * np.pi * i / golden
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pts[i] = [x * 0.45, y * 0.45, z * 0.45]
        return pts.flatten()
    
    starts = [
        init_grid_5x5x4(),
        init_grid_4x5x5(),
        init_grid_5x4x5(),
        init_grid_5x5x4() + np.random.normal(0, 0.01, size=(N*3,)),
        init_grid_4x5x5() + np.random.normal(0, 0.01, size=(N*3,)),
        init_grid_5x4x5() + np.random.normal(0, 0.01, size=(N*3,)),
        init_random(),
        init_random(),
        init_fibonacci(),
    ]
    
    # Clip starts to valid range
    for i in range(len(starts)):
        starts[i] = np.clip(starts[i], -0.49, 0.49)
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Try different power values for the soft-min approximation
    P_values = [20.0, 30.0, 40.0]
    
    for P in P_values:
        if time.time() > deadline - 10:
            break
            
        for i, x0 in enumerate(starts):
            if time.time() > deadline - 5:
                break
                
            try:
                res = minimize(lambda x: objective_and_grad(x, P), x0, 
                             method='L-BFGS-B', bounds=bounds, 
                             jac=True,
                             options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
                if res.fun < 1e15:
                    pts = res.x.reshape(N, 3)
                    pts = np.clip(pts, -0.5, 0.5)
                    score = calculate_score(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
            except Exception:
                continue
    
    # Local refinement with higher precision
    if best_points is not None and time.time() < deadline - 15:
        for P in [30.0, 40.0]:
            if time.time() > deadline - 10:
                break
            try:
                x0 = best_points.flatten()
                res = minimize(lambda x: objective_and_grad(x, P), x0, 
                             method='L-BFGS-B', bounds=bounds, 
                             jac=True,
                             options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 30})
                if res.fun < 1e15:
                    pts = res.x.reshape(N, 3)
                    pts = np.clip(pts, -0.5, 0.5)
                    score = calculate_score(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
            except Exception:
                break
    
    # Final local moves: try to nudge points to improve minimum distance
    if best_points is not None and time.time() < deadline - 5:
        pts = best_points.copy()
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            # Find the pair with minimum distance
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = min_idx[0], min_idx[1]
            min_d = dists[i, j]
            
            # Nudge both points slightly apart
            direction = pts[i] - pts[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
                # Small nudge
                step = 0.001
                pts[i] += direction * step
                pts[j] -= direction * step
                # Clip to bounds
                pts[i] = np.clip(pts[i], -0.499, 0.499)
                pts[j] = np.clip(pts[j], -0.499, 0.499)
            
            score = calculate_score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
            else:
                pts = best_points.copy()
    
    if best_points is None:
        best_points = init_grid_5x5x4().reshape(N, 3)
        
    return best_points
# EVOLVE-BLOCK-END
