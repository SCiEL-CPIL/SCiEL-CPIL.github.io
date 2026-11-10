# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    def soft_min(dists, t=10.0):
        """
        Differentiable approximation of the minimum distance.
        t controls the sharpness of the approximation.
        """
        # Scale to avoid overflow/underflow in exp
        scaled = dists / t
        # Shift to ensure numerical stability (subtract max)
        max_val = np.max(scaled)
        # If all values are very large negative, max_val might be -inf or similar, but dists are positive
        # Actually, for soft min: -t * log( sum( exp(-dists/t) ) / N )
        # Let u = -dists/t. We want -t * log( sum(exp(u))/N )
        # u_max = max(u) = -min(dists)/t
        # sum(exp(u)) = exp(u_max) * sum(exp(u - u_max))
        # log(sum(exp(u))/N) = u_max + log(sum(exp(u - u_max))/N)
        # result = -t * (u_max + log(sum(exp(u - u_max))/N))
        
        u = -dists / t
        u_max = np.max(u)
        # Handle case where u_max is -inf (should not happen for finite dists)
        if not np.isfinite(u_max):
            return 0.0
            
        exp_sum = np.sum(np.exp(u - u_max))
        if exp_sum == 0:
            return 0.0
            
        log_term = u_max + np.log(exp_sum / len(dists))
        return -t * log_term

    def objective(x):
        """
        Objective function to minimize.
        We want to maximize min_dist / max_dist.
        By normalizing points so max_dist = 1, we just maximize min_dist.
        We approximate min_dist with soft_min.
        """
        pts = x.reshape(n, d)
        
        # Calculate pairwise distances
        dists = pdist(pts)
        
        # Find max distance
        d_max = np.max(dists)
        if d_max < 1e-12:
            # All points are coincident or nearly so
            return 0.0
            
        # Normalize distances so that d_max = 1
        # This makes the problem scale-invariant and bounded
        dists_norm = dists / d_max
        
        # We want to maximize the minimum of dists_norm
        # So we minimize the negative of the soft-min
        min_dist_soft = soft_min(dists_norm, t=10.0)
        return -min_dist_soft

    def get_score(pts):
        """Calculate the actual score R = (d_min / d_max)^2"""
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Initial guess: 4x4 grid
    x0 = np.zeros((n, d))
    for i in range(n):
        x0[i, 0] = i % 4
        x0[i, 1] = i // 4
    
    # Center the initial guess
    x0 = x0 - np.mean(x0, axis=0)
    
    # Flatten for optimizer
    x0_flat = x0.flatten()

    best_x = x0_flat
    best_score = get_score(x0)

    # Run optimization
    try:
        # Use L-BFGS-B
        # maxiter and maxfun help prevent running too long
        res = minimize(
            objective, 
            x0_flat, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12}
        )
        
        if res.success or res.fun < objective(best_x):
            # Check time before accepting
            if time.time() < deadline:
                candidate_x = res.x
                candidate_pts = candidate_x.reshape(n, d)
                candidate_score = get_score(candidate_pts)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_x = candidate_x
    except Exception as e:
        # If optimization fails, stick with best found so far
        pass

    # If we still have time, we could try a second run with different initialization or parameters
    # But for now, let's just return the best found.
    # A simple restart with random initialization might help escape local minima
    
    if time.time() < deadline - 60:
        # Try a few random restarts
        np.random.seed(42)
        for i in range(5):
            if time.time() > deadline - 30:
                break
            x_rand = np.random.randn(n, d)
            x_rand = x_rand - np.mean(x_rand, axis=0)
            x_rand_flat = x_rand.flatten()
            
            try:
                res = minimize(
                    objective, 
                    x_rand_flat, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12}
                )
                if time.time() < deadline:
                    candidate_x = res.x
                    candidate_pts = candidate_x.reshape(n, d)
                    candidate_score = get_score(candidate_pts)
                    
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_x = candidate_x
            except:
                pass

    # Return the best points found
    best_points = best_x.reshape(n, d)
    
    # Ensure it's a numpy array of floats
    return best_points.astype(np.float64)

# EVOLVE-BLOCK-END
