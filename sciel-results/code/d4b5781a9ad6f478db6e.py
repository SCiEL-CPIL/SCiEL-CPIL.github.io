import numpy as np
from scipy.optimize import minimize

N = 14
D = 3

def construct():
    # Initial guess for the ratio b/a
    # We found b/a ~ 1.73 gives ratio ~ 4.73
    # Let's start with b = 1.73 * a. Set a=1.
    x0 = np.array([1.0, 1.73])
    
    def objective(params):
        a, b = params
        if a <= 0:
            return 1e10
        # Generate points
        # Cube vertices: (±a, ±a, ±a)
        cube_verts = np.array(list(np.meshgrid([-a, a], [-a, a], [-a, a]))).T
        # Face centers: (±b, 0, 0), (0, ±b, 0), (0, 0, ±b)
        face_centers = np.array([
            [b, 0, 0], [-b, 0, 0],
            [0, b, 0], [0, -b, 0],
            [0, 0, b], [0, 0, -b]
        ])
        
        points = np.vstack([cube_verts, face_centers])
        
        # Calculate pairwise distances squared
        # Use broadcasting
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Exclude self-distances
        np.fill_diagonal(dist_sq, np.inf)
        
        min_d_sq = np.min(dist_sq)
        max_d_sq = np.max(dist_sq)
        
        if min_d_sq < 1e-9:
            return 1e10
            
        return max_d_sq / min_d_sq

    # Optimize
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    a_opt, b_opt = res.x
    
    # Reconstruct points with optimal parameters
    cube_verts = np.array(list(np.meshgrid([-a_opt, a_opt], [-a_opt, a_opt], [-a_opt, a_opt]))).T
    face_centers = np.array([
        [b_opt, 0, 0], [-b_opt, 0, 0],
        [0, b_opt, 0], [0, -b_opt, 0],
        [0, 0, b_opt], [0, 0, -b_opt]
    ])
    
    points = np.vstack([cube_verts, face_centers])
    
    return points
