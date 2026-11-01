import numpy as np
from scipy.optimize import minimize

def construct():
    N = 50
    # Initialize with a 4x4x4 grid subset
    ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Select 50 points. To be deterministic but not biased, we can use a fixed permutation
    # or just the first 50 if we sort them in a way that spreads them out.
    # The current code uses np.random.permutation which is non-deterministic across runs?
    # np.random.seed is not set. We should set it for reproducibility.
    np.random.seed(42)
    idx = np.random.permutation(64)[:50]
    centers = grid[idx].copy()
    
    # Add small jitter to break symmetry
    centers += np.random.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    def objective(x):
        """
        Minimizes the negative of the smooth approximation of the minimum radius.
        x: shape (N, 3)
        """
        c = x.reshape(N, 3)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Wall distances squared
        # d_wall_i = 0.5 - max(|x_i|, |y_i|, |z_i|)
        # We want to maximize min(d_wall).
        # Smooth min approximation: -1/tau * log(sum(exp(-tau * d)))
        # Here we work with squared distances for gradient stability?
        # Let's stick to distances.
        
        # Compute distances
        dist = np.sqrt(dist_sq)
        # Wall distances
        abs_c = np.abs(c)
        max_coord = np.max(abs_c, axis=1)
        wall_dists = 0.5 - max_coord
        
        # Combine all distances into one array
        # We want the min of all pair distances and wall distances.
        # Flatten pair distances (upper triangle)
        iu = np.triu_indices(N, k=1)
        pair_dists = dist[iu]
        
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        # Smooth minimum approximation
        # min(x) approx -1/tau * log(sum(exp(-tau * x)))
        # We want to maximize min, so minimize -min.
        # -min(x) approx 1/tau * log(sum(exp(-tau * x)))
        
        tau = 50.0 # Temperature parameter. Higher is sharper.
        
        # Numerical stability: subtract max
        # exp(-tau * x) can underflow if x is large or tau is large.
        # But x is around 0.1-0.2. tau=50 -> tau*x ~ 5-10. exp(-10) is fine.
        
        # To avoid overflow in exp, we can do:
        # log(sum(exp(-tau * x))) = -tau * max(x) + log(sum(exp(-tau * (x - max(x)))))
        # But we want the MIN.
        # Let y = -x. Max(y) = -Min(x).
        # Min(x) = -Max(-x).
        # -Min(x) = Max(-x).
        # Max(y) approx 1/tau * log(sum(exp(tau * y)))
        
        neg_dists = -all_dists
        max_neg = np.max(neg_dists)
        
        # log-sum-exp
        exp_terms = np.exp(tau * (neg_dists - max_neg))
        log_sum = np.log(np.sum(exp_terms))
        
        smooth_min_approx = max_neg + log_sum / tau
        
        # We want to maximize the min distance, so minimize -smooth_min_approx
        return -smooth_min_approx
    
    # Optimize
    # Use L-BFGS-B
    result = minimize(objective, centers.ravel(), method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
    
    centers = result.x.reshape(N, 3)
    
    # Final clamp
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
