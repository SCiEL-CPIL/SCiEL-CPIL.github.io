import numpy as np
from scipy.optimize import minimize

def construct():
    N = 50
    
    # Objective function to minimize (negative score)
    def objective(params):
        # params: 100 angles (theta, phi interleaved or separate) + 1 R
        # Let's use 2*N for angles, 1 for R
        angles = params[:2*N]
        R = params[2*N]
        
        # Ensure R is within valid bounds
        # R must be < 0.5 for wall distance > 0
        # R must be > 0
        if R <= 0 or R >= 0.5:
            return 1.0 # Large penalty (minimizing -score, so large score is bad? No, we minimize -r. If r is small, -r is large positive? No.
            # We want to maximize r. So we minimize -r.
            # If invalid, return a large positive number to indicate badness?
            # Or just return 0? 
            # Let's return a large value to push optimizer away.
            return 100.0
        
        theta = angles[0::2]
        phi = angles[1::2]
        
        # Convert to Cartesian
        # x = R sin(theta) cos(phi)
        # y = R sin(theta) sin(phi)
        # z = R cos(theta)
        
        # Clip theta to [0, pi] to avoid issues? 
        # Actually, theta and phi are unbounded in params, but sin/cos handle it.
        # However, for unique representation, we might want to constrain them.
        # Nelder-Mead doesn't handle bounds well.
        # We can use a transformation: theta = pi * sigmoid(t)? 
        # Or just let it wander. sin/cos are periodic.
        
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        
        x = R * sin_t * cos_p
        y = R * sin_t * sin_p
        z = R * cos_t
        
        centers = np.column_stack((x, y, z))
        
        # Pairwise distances
        # dist[i, j] = ||c_i - c_j||
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Min pairwise distance
        # Exclude diagonal
        dist_mask = ~np.eye(N, dtype=bool)
        min_pair_dist = np.min(dist[dist_mask])
        
        # Wall distance
        # Since centers are on sphere of radius R centered at 0,
        # max |c_ik| is R.
        # min_k (0.5 - |c_ik|) = 0.5 - R.
        wall_dist = 0.5 - R
        
        r = min(min_pair_dist / 2.0, wall_dist)
        
        return -r

    # Initial guess
    # Fibonacci sphere for uniform distribution
    def fibonacci_sphere(n):
        # Returns theta, phi
        i = np.arange(n)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        theta = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
        phi = golden_angle * i
        return theta, phi

    best_score = -np.inf
    best_params = None
    
    # Run a few restarts
    num_restarts = 10
    for k in range(num_restarts):
        # Random initialization of angles
        # Or use Fibonacci + noise
        theta0, phi0 = fibonacci_sphere(N)
        
        # Add random noise
        theta0 = theta0 + np.random.normal(0, 0.1, N)
        phi0 = phi0 + np.random.normal(0, 0.1, N)
        
        # Initial R
        R0 = 0.4 + np.random.uniform(-0.05, 0.05)
        
        # Flatten params
        angles0 = np.empty(2*N)
        angles0[0::2] = theta0
        angles0[1::2] = phi0
        params0 = np.concatenate([angles0, [R0]])
        
        # Optimize
        # Use Powell for robustness with non-smooth objective
        res = minimize(objective, params0, method='Powell', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x

    # Extract best params
    if best_params is None:
        # Fallback to a simple grid or something if optimization failed completely
        # But Powell should always return something
        theta0, phi0 = fibonacci_sphere(N)
        R0 = 0.4
        angles0 = np.empty(2*N)
        angles0[0::2] = theta0
        angles0[1::2] = phi0
        best_params = np.concatenate([angles0, [R0]])
        best_score = -objective(best_params)

    # Reconstruct centers from best_params
    angles = best_params[:2*N]
    R = best_params[2*N]
    
    theta = angles[0::2]
    phi = angles[1::2]
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    x = R * sin_t * cos_p
    y = R * sin_t * sin_p
    z = R * cos_t
    
    centers = np.column_stack((x, y, z))
    
    return centers
