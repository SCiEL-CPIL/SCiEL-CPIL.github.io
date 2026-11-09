import numpy as np
from scipy.optimize import minimize

def construct():
    # Initial guess: Regular Octagon
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    # Scale to diameter 2 (regular octagon side length s, diameter d = s * sqrt(4+2sqrt(2)) ... 
    # Actually, for regular n-gon, diameter is distance between opposite vertices = 2 * radius.
    # So if radius is 1, diameter is 2.
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack((x, y))
    
    # Flatten for optimization
    x0 = points.flatten()
    
    def objective(flat_points):
        pts = flat_points.reshape(8, 2)
        # Calculate all pairwise distances
        # We want to maximize product of distances, so minimize -log(product)
        # First, scale to diameter 2
        # Diameter is max distance
        # Using scipy.spatial.distance.pdist for efficiency
        from scipy.spatial.distance import pdist
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 1e10 # Penalty for collapse
        # Scale
        pts_scaled = pts * (2.0 / d_max)
        dists_scaled = dists * (2.0 / d_max)
        
        # Product of all pairwise distances
        # log(product) = sum(log(dists))
        # If any dist is 0, log is -inf. Add small epsilon.
        log_prod = np.sum(np.log(dists_scaled + 1e-12))
        
        return -log_prod

    # Use Nelder-Mead as it doesn't require gradients and handles the non-smooth scaling well
    # L-BFGS-B might get stuck if the scaling introduces non-differentiability at the max distance pair changes
    # Nelder-Mead is robust for low dimensions (16D)
    
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    opt_points = result.x.reshape(8, 2)
    
    # Final scaling to ensure diameter is exactly 2
    from scipy.spatial.distance import pdist
    dists = pdist(opt_points)
    d_max = np.max(dists)
    if d_max > 1e-12:
        opt_points = opt_points * (2.0 / d_max)
        
    # Return as list of lists of floats
    return opt_points.tolist()
