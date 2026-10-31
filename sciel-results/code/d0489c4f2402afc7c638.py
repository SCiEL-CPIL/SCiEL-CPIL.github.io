import numpy as np
from scipy.optimize import minimize

def construct():
    N = 16
    d = 2
    
    def objective(params):
        # Reshape params to (N, 2)
        pts = params.reshape(N, d)
        
        # Center the points
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        
        # Scale points to have fixed sum of squared norms to break scale invariance
        # If all points are zero, return large value
        ssd = np.sum(pts**2)
        if ssd < 1e-12:
            return 1e10
        pts = pts * np.sqrt(1.0 / ssd)
        
        # Calculate all pairwise squared distances
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dist_sq_masked = dist_sq[mask]
        
        min_dist_sq = np.min(dist_sq_masked)
        max_dist_sq = np.max(dist_sq_masked)
        
        if min_dist_sq < 1e-12:
            return 1e10
            
        return max_dist_sq / min_dist_sq

    # Initial guess: points on a circle
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_pts = np.column_stack([np.cos(theta), np.sin(theta)])
    init_params = init_pts.flatten()
    
    # Try multiple random starts to escape local minima
    best_params = init_params
    best_score = objective(init_params)
    
    # Run a few local optimizations from different starts
    for i in range(5):
        if i > 0:
            # Perturb the initial circle
            noise = np.random.normal(0, 0.5, init_params.shape)
            curr_params = init_params + noise
        else:
            curr_params = init_params
            
        res = minimize(objective, curr_params, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    
    # Reconstruct the best points
    best_pts = best_params.reshape(N, d)
    centroid = np.mean(best_pts, axis=0)
    best_pts = best_pts - centroid
    
    return best_pts
