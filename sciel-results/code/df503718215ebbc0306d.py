# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates.

    """
    n = 16

    def objective(flat_points):
        pts = flat_points.reshape(n, 2)
        
        # Center
        pts = pts - np.mean(pts, axis=0)
        
        # Calculate max distance to normalize scale
        dists = pdist(pts)
        d_max = np.max(dists)
        
        if d_max < 1e-10:
            return 0.0
            
        # Scale so that d_max = 1
        pts = pts / d_max
        
        # Recalculate distances
        dists = pdist(pts)
        d_min = np.min(dists)
        
        # We want to maximize d_min, so minimize -d_min
        return -d_min

    def generate_initial_configs():
        configs = []
        
        # 1. Random configurations
        rng = np.random.default_rng(42)
        for _ in range(10):
            pts = rng.uniform(-1, 1, size=(n, 2))
            configs.append(pts.flatten())
            
        # 2. Grid configuration (4x4)
        x = np.linspace(-1, 1, 4)
        y = np.linspace(-1, 1, 4)
        xx, yy = np.meshgrid(x, y)
        grid_pts = np.column_stack((xx.flatten(), yy.flatten()))
        configs.append(grid_pts.flatten())
        
        # 3. Hexagonal-like configuration
        # Approximate 16 points in a hexagonal packing
        # Center + 6 + 9? 
        # Let's just use a distorted grid or random perturbation of grid
        for i in range(5):
            noise = rng.normal(0, 0.2, size=(n, 2))
            hex_pts = grid_pts + noise
            configs.append(hex_pts.flatten())
            
        return configs

    best_result = None
    best_score = -np.inf

    # Run optimization from multiple starts
    for init_config in generate_initial_configs():
        # Use Nelder-Mead for robustness on non-smooth objectives
        # maxiter and options to allow enough time
        res = minimize(
            objective, 
            init_config, 
            method='Nelder-Mead', 
            options={
                'maxiter': 2000, 
                'xatol': 1e-8, 
                'fatol': 1e-10
            }
        )
        
        score = -res.fun # Since we minimized -d_min, the score is d_min
        if score > best_score:
            best_score = score
            best_result = res.x

    best_points = best_result.reshape(n, 2)
    
    # Final normalization
    best_points -= np.mean(best_points, axis=0)
    dists = pdist(best_points)
    d_max = np.max(dists)
    if d_max > 1e-10:
        best_points = best_points / d_max
        
    return best_points

# EVOLVE-BLOCK-END
