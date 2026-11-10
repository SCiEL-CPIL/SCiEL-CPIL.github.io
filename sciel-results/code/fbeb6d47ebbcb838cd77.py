import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initialization
    def get_initial_points(n):
        spacing = 0.05
        while True:
            k = int(1.0 / spacing) + 5
            pts = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if x*x + y*y <= 1.0:
                        pts.append([x, y])
            pts = np.array(pts)
            if len(pts) >= n:
                np.random.shuffle(pts)
                return pts[:n]
            spacing *= 0.95
            if spacing < 0.01: break
        return np.random.rand(n, 2) * 2 - 1

    points = get_initial_points(N).flatten()
    
    # Project to disk
    def project_to_disk(pts_flat):
        pts = pts_flat.reshape(-1, 2)
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] /= norms[mask, np.newaxis]
        return pts.flatten()

    points = project_to_disk(points)

    def objective_and_grad(x):
        pts = x.reshape(-1, 2)
        
        # Distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        # Potential U = sum 1/d^2
        # We sum over i<j, so divide by 2
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        U = 0.5 * np.sum(inv_dist_sq)
        
        # Gradient of U
        # dU/dx_i = sum_{j!=i} -2 (x_i - x_j) / d^4
        # Note: diff[i,j] = x_i - x_j
        # term = -2 * diff / dist_sq^2
        grad_term = -2.0 * diff / (dist_sq**2)[:, :, np.newaxis]
        np.fill_diagonal(grad_term, 0.0)
        grad_U = np.sum(grad_term, axis=1)
        
        # Boundary Penalty
        # P = K * sum (max(0, r_i - 1))^2
        norms = np.linalg.norm(pts, axis=1)
        violation = np.maximum(0.0, norms - 1.0)
        P = 100.0 * np.sum(violation**2)
        
        # Gradient of P
        # dP/dx_i = 200 * (r_i - 1) * (x_i / r_i) if r_i > 1 else 0
        grad_P = np.zeros_like(pts)
        mask = norms > 1.0
        if np.any(mask):
            grad_P[mask] = 200.0 * (norms[mask] - 1.0) * pts[mask] / norms[mask, np.newaxis]
            
        grad = (grad_U + grad_P).flatten()
        
        return U + P, grad

    best_score = -1.0
    best_points = points.copy()
    
    # Run optimization
    # L-BFGS-B is good
    # We can run multiple iterations or just one long one
    # Max iterations depends on speed. Let's try 500-1000.
    
    try:
        res = minimize(objective_and_grad, points, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 1000, 'gtol': 1e-5, 'maxls': 20})
        
        if res.success or res.fun < 1e10: # Check if it didn't explode
            final_points = project_to_disk(res.x)
            
            # Calculate score
            pts = final_points.reshape(-1, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            min_pair_dist = np.sqrt(np.min(dist_sq))
            
            norms = np.linalg.norm(pts, axis=1)
            min_boundary_dist = 1.0 - np.max(norms)
            
            r = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if r > best_score:
                best_score = r
                best_points = final_points.copy()
                
    except Exception:
        pass

    # If time remains, try a few more restarts with different seeds
    while time.time() < deadline - 10.0:
        # New init
        np.random.seed(int(time.time()*1000) % 10000)
        points = get_initial_points(N).flatten()
        points = project_to_disk(points)
        
        try:
            res = minimize(objective_and_grad, points, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 500, 'gtol': 1e-4})
            
            if res.success or res.fun < 1e10:
                final_points = project_to_disk(res.x)
                pts = final_points.reshape(-1, 2)
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dist_sq = np.sum(diff**2, axis=2)
                np.fill_diagonal(dist_sq, np.inf)
                min_pair_dist = np.sqrt(np.min(dist_sq))
                
                norms = np.linalg.norm(pts, axis=1)
                min_boundary_dist = 1.0 - np.max(norms)
                
                r = min(min_pair_dist / 2.0, min_boundary_dist)
                
                if r > best_score:
                    best_score = r
                    best_points = final_points.copy()
        except Exception:
            break

    return best_points.reshape(-1, 2)
