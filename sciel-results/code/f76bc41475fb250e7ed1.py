"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _cartesian_to_spherical(p):
    z = p[:, 2]
    # Clip z to [-1, 1] for numerical stability in arccos
    z_clipped = np.clip(z, -1.0, 1.0)
    theta = np.arccos(z_clipped)
    phi = np.arctan2(p[:, 1], p[:, 0])
    return theta, phi

def _compute_covering_radius(points, num_samples=20000):
    """
    Estimate the covering radius by sampling the sphere.
    """
    # Generate random points on the sphere
    # Use Gaussian distribution and normalize
    samples = np.random.normal(0, 1, (num_samples, 3))
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    # Compute distances from samples to all points
    # dist = arccos(dot)
    # dot = samples @ points.T
    dots = samples @ points.T
    # Clip for stability
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # Find min distance for each sample
    min_dists = np.min(dists, axis=1)
    
    # Covering radius is the max of these min distances
    return np.max(min_dists)

def construct():
    best_points = None
    best_radius = np.inf
    num_restarts = 5
    
    # Beta parameter for softmin approximation
    # Higher beta makes the function closer to the true min, but harder to optimize
    beta = 15.0
    
    for restart in range(num_restarts):
        # 1. Generate initial points
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        
        if restart == 0:
            p_cart = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Add small random noise to perturb the Fibonacci spiral
            # Perturb in Cartesian space then project back? Or perturb spherical coords?
            # Easier: Perturb spherical coordinates
            theta_init = np.arccos(z)
            phi_init = np.arctan2(r * np.sin(phi), r * np.cos(phi))
            
            # Add noise
            noise_scale = 0.1 + 0.1 * restart
            theta_init += np.random.normal(0, noise_scale, N)
            phi_init += np.random.normal(0, noise_scale, N)
            
            # Clamp theta to valid range [0, pi]
            theta_init = np.clip(theta_init, 0.01, np.pi - 0.01)
            
            p_cart = _spherical_to_cartesian(theta_init, phi_init)
            # Normalize
            p_cart /= np.linalg.norm(p_cart, axis=1, keepdims=True)
        
        theta, phi_sph = _cartesian_to_spherical(p_cart)
        x0 = np.concatenate([theta, phi_sph])
        
        # 2. Define the cost function: Softmin of pairwise distances
        # We want to MAXIMIZE the minimum distance.
        # So we MINIMIZE the negative softmin.
        # softmin(x) = -1/beta * ln(sum(exp(-beta * x)))
        # cost = -softmin(x) = 1/beta * ln(sum(exp(-beta * x)))
        
        def softmin_cost(params):
            theta = params[:N]
            phi_sph = params[N:]
            
            p = _spherical_to_cartesian(theta, phi_sph)
            # Normalize to handle any numerical drift
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            
            # Compute pairwise dot products
            dots = p @ p.T
            # Clip for numerical stability
            dots = np.clip(dots, -1.0, 1.0)
            
            # Compute pairwise angular distances
            dists = np.arccos(dots)
            
            # We only care about i < j
            # Create a mask for upper triangle
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            dists_upper = dists[mask]
            
            if len(dists_upper) == 0:
                return 0.0
            
            # Compute softmin
            # exp(-beta * dists)
            # If beta is large, this can underflow/overflow. 
            # Use log-sum-exp trick for stability:
            # ln(sum(exp(-beta * d))) = -beta * max(d) + ln(sum(exp(-beta * (d - max_d))))
            max_d = np.max(dists_upper)
            exp_terms = np.exp(-beta * (dists_upper - max_d))
            sum_exp = np.sum(exp_terms)
            
            if sum_exp == 0:
                return 0.0
            
            log_sum_exp = -beta * max_d + np.log(sum_exp)
            softmin_val = -1.0 / beta * log_sum_exp
            
            # We want to minimize the negative of the softmin? 
            # No, softmin approximates min.
            # min(dist) is what we want to maximize.
            # So we want to minimize -min(dist).
            # cost = -softmin_val
            return -softmin_val

        # 3. Optimize
        try:
            res = minimize(softmin_cost, x0, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < best_radius: # Check if result is valid
                theta_opt = res.x[:N]
                phi_opt = res.x[N:]
                p_opt = _spherical_to_cartesian(theta_opt, phi_opt)
                p_opt /= np.linalg.norm(p_opt, axis=1, keepdims=True)
                
                # Evaluate true covering radius
                # Use a fixed seed for reproducibility in evaluation? 
                # The evaluator calls construct() once. 
                # We should use a random seed internally or just accept the stochastic nature.
                # To be safe, we use a random seed for the samples.
                np.random.seed(42) # Fixed seed for radius estimation to be consistent
                radius = _compute_covering_radius(p_opt, num_samples=50000)
                
                if radius < best_radius:
                    best_radius = radius
                    best_points = p_opt
                    
        except Exception as e:
            continue
            
    # If no valid point found (should not happen), return Fibonacci
    if best_points is None:
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_points
# EVOLVE-BLOCK-END
