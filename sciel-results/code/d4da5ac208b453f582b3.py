import numpy as np
from scipy.optimize import minimize

def construct():
    N = 50
    
    # Fibonacci sphere initialization
    def fibonacci_sphere(n):
        points = []
        phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
        theta = np.pi * (1 + 5 ** 0.5) * np.arange(n)
        for i in range(n):
            x = np.sin(phi[i]) * np.cos(theta[i])
            y = np.sin(phi[i]) * np.sin(theta[i])
            z = np.cos(phi[i])
            points.append([x, y, z])
        return np.array(points)
    
    # Initial angular coordinates
    # We store theta (polar) and phi (azimuthal)
    # Or just store the unit vectors and optimize them?
    # Optimizing unit vectors directly is hard due to constraint ||v||=1.
    # Using angles is better.
    
    unit_pts = fibonacci_sphere(N)
    # Convert to spherical coords
    # theta = arccos(z), phi = arctan2(y, x)
    thetas = np.arccos(np.clip(unit_pts[:, 2], -1.0, 1.0))
    phis = np.arctan2(unit_pts[:, 1], unit_pts[:, 0])
    
    # Flatten for optimizer
    # Variables: [theta_0, ..., theta_N-1, phi_0, ..., phi_N-1, R]
    # Total 2N + 1 variables.
    
    def unpack(x):
        thetas = x[:N]
        phis = x[N:2*N]
        R = x[2*N]
        return thetas, phis, R

    def objective(x):
        thetas, phis, R = unpack(x)
        if R <= 0 or R > 0.5:
            return 1e6 # Penalty for invalid R
        
        # Calculate positions
        # x = R * sin(theta) * cos(phi)
        # y = R * sin(theta) * sin(phi)
        # z = R * cos(theta)
        sin_t = np.sin(thetas)
        cos_t = np.cos(thetas)
        cos_p = np.cos(phis)
        sin_p = np.sin(phis)
        
        xs = R * sin_t * cos_p
        ys = R * sin_t * sin_p
        zs = R * cos_t
        
        # Wall distances
        # dist_wall_i = 0.5 - max(|xs_i|, |ys_i|, |zs_i|)
        max_coords = np.max(np.abs(np.stack([xs, ys, zs], axis=1)), axis=1)
        wall_dists = 0.5 - max_coords
        
        # Pairwise distances
        # We need min over i<j of dist_ij
        # dist_ij = sqrt((xi-xj)^2 + ...)
        # To avoid O(N^2) loop in Python, use numpy broadcasting
        # But N=50, so 1225 pairs. It's fine to loop or use broadcasting.
        
        diff_x = xs[:, None] - xs[None, :]
        diff_y = ys[:, None] - ys[None, :]
        diff_z = zs[:, None] - zs[None, :]
        
        dists = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        min_pair_dist = np.min(dists[mask])
        
        # Current radius
        r = min(min_pair_dist / 2.0, np.min(wall_dists))
        
        # We want to maximize r, so minimize -r
        return -r

    # Initial guess for R
    # If we pack N points on a sphere, the angular spacing is approx sqrt(4 pi / N)
    # Chord length approx R * angular_spacing
    # We want R * angular_spacing / 2 ~ 0.5 - R
    # Let's start with R = 0.3
    
    x0 = np.concatenate([thetas, phis, [0.3]])
    
    # Bounds
    # theta in [0, pi], phi in [-pi, pi], R in [0.1, 0.5]
    bounds = [(0, np.pi)] * N + [(-np.pi, np.pi)] * N + [(0.1, 0.5)]
    
    # Use Nelder-Mead
    # maxiter needs to be high enough
    res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                   options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    
    # Extract best solution
    best_x = res.x
    thetas, phis, R = unpack(best_x)
    
    sin_t = np.sin(thetas)
    cos_t = np.cos(thetas)
    cos_p = np.cos(phis)
    sin_p = np.sin(phis)
    
    centers = np.column_stack([
        R * sin_t * cos_p,
        R * sin_t * sin_p,
        R * cos_t
    ])
    
    return centers
