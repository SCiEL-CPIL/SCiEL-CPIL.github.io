import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    # Define the 14 points based on Tetrahedral symmetry
    # Orbit 1: 4 Vertices of regular tetrahedron
    # Directions: (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)
    v_dirs = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) / np.sqrt(3)
    
    # Orbit 2: 6 Edge Midpoints (axes)
    # Directions: (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)
    e_dirs = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ])
    
    # Orbit 3: 4 Face Centers (dual tetrahedron)
    # Directions: (1,1,-1), (1,-1,1), (-1,1,1), (-1,-1,-1)
    f_dirs = np.array([
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [-1, -1, -1]
    ]) / np.sqrt(3)
    
    # Initial guess: all radii 1.0
    # Order: [r_v, r_e, r_f]
    x0 = np.array([1.0, 1.0, 1.0])
    
    def objective(params):
        r_v, r_e, r_f = params
        # Ensure positive radii
        if r_v <= 0 or r_e <= 0 or r_f <= 0:
            return 1e6
            
        points_v = v_dirs * r_v
        points_e = e_dirs * r_e
        points_f = f_dirs * r_f
        
        points = np.vstack([points_v, points_e, points_f])
        
        # Calculate distances
        d = pdist(points)
        if len(d) == 0:
            return 1e6
        d_min = np.min(d)
        d_max = np.max(d)
        
        if d_max == 0:
            return 1e6
            
        ratio = (d_min / d_max) ** 2
        # We want to maximize ratio, so minimize -log(ratio) or just -ratio
        # Using -ratio is fine, but log is more stable for small ratios
        return -np.log(ratio + 1e-12)

    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    best_params = result.x
    r_v, r_e, r_f = best_params
    
    points_v = v_dirs * r_v
    points_e = e_dirs * r_e
    points_f = f_dirs * r_f
    
    points = np.vstack([points_v, points_e, points_f])
    
    return points
