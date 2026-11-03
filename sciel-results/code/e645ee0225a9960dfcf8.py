"""Improved program for the suite task; uses smooth optimization with hexagonal init."""
import numpy as np
from scipy.optimize import minimize

N = 50


def construct():
    # 1. Initialization: Hexagonal Lattice
    # Generate a hexagonal grid that covers the unit square
    # Spacing estimated for N=50
    # Area per point ~ 1/50 = 0.02
    # Hex area = s^2 * sqrt(3)/2 => s ~ sqrt(0.02 * 2 / sqrt(3)) ~ 0.15
    
    s = 0.15
    # Generate points in a hex grid
    points = []
    # Range to cover [-0.5, 0.5]
    range_x = np.arange(-0.8, 0.8, s)
    range_y = np.arange(-0.8, 0.8, s)
    
    for i, x in enumerate(range_x):
        # Offset every other row
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
            
    points = np.array(points)
    
    # Keep points inside the square (with margin to allow optimization)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    
    # If we have more than N, randomly select N (or select closest to grid center)
    # To ensure good distribution, we can just take the first N after shuffling
    # Or better, select points that are most "central" to avoid edge effects in init
    if len(points) > N:
        # Calculate distance from center
        dists = np.sum(points**2, axis=1)
        # Sort by distance and take first N? No, that clusters in center.
        # Random selection is better for diversity, but deterministic is better for reproducibility.
        # Let's take a random subset with a fixed seed
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
        
    # If we have fewer (unlikely), pad with random points
    while len(points) < N:
        points = np.vstack([points, np.random.uniform(-0.4, 0.4, 2)])
        
    initial_centers = points[:N]
    
    # 2. Optimization
    # We want to maximize r = min( min_{i<j} d_ij/2, min_i w_i )
    # This is equivalent to minimizing -r.
    # We use a smooth approximation of the min function:
    # min(a, b) approx -1/k * log(exp(-k*a) + exp(-k*b))
    # Here we have many terms.
    # Objective: Minimize - (1/k) * log( sum_{i<j} exp(-k * d_ij) + sum_i exp(-k * w_i) )
    # Note: The radius is limited by d_ij/2. So we should use d_ij/2 in the exponent?
    # Let's define "effective distance" for pairs as d_ij/2 and for walls as w_i.
    # Then r is the min of these effective distances.
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def objective(params, k):
        centers = params.reshape(N, 2)
        
        # Pairwise distances
        # d_ij
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Zero out diagonal
        np.fill_diagonal(dists, np.inf)
        
        # Effective pair distance is d_ij / 2
        pair_terms = dists / 2.0
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        # Effective wall distance is w_i
        
        # Combine all constraints
        # We want to maximize the minimum of all pair_terms and wall_dists.
        # Let's flatten them
        all_terms = np.concatenate([pair_terms.flatten(), wall_dists.flatten()])
        
        # Filter out inf
        all_terms = all_terms[np.isfinite(all_terms)]
        
        if len(all_terms) == 0:
            return 0.0
            
        # Log-sum-exp trick for numerical stability
        # min(x) approx -1/k * log(sum(exp(-k*x)))
        # To avoid underflow, subtract max
        # Actually, we want to MINIMIZE -min(x).
        # So we maximize min(x).
        # max_x min(x) is equivalent to minimizing -min(x).
        
        # Let's compute the soft-min
        # soft_min = -1/k * log( sum( exp(-k * x) ) )
        
        # For stability:
        # log( sum( exp(-k*x) ) ) = -k * max(x) + log( sum( exp(-k*(x - max(x))) ) )
        # Wait, the standard trick is for log-sum-exp of positive numbers.
        # Here we have exp(-k*x). If x is large, exp(-k*x) is small.
        # If x is small (close to 0), exp(-k*x) is large.
        # The term that dominates the sum is the one with the smallest x (largest exponent value if k>0? No).
        # Let's check: x1=0.1, x2=0.2. k=10.
        # exp(-1) = 0.36, exp(-2) = 0.13. Sum = 0.49.
        # -1/10 * log(0.49) = -1/10 * (-0.71) = 0.071.
        # True min is 0.1.
        # As k increases, soft_min approaches min.
        
        # To compute stably:
        # Let m = min(all_terms).
        # sum(exp(-k*x)) = exp(-k*m) * sum(exp(-k*(x-m)))
        # log(sum(exp(-k*x))) = -k*m + log(sum(exp(-k*(x-m))))
        # soft_min = -1/k * ( -k*m + log(sum(exp(-k*(x-m)))) )
        # soft_min = m - 1/k * log(sum(exp(-k*(x-m))))
        
        m = np.min(all_terms)
        # x - m >= 0, so -k*(x-m) <= 0. exp(-k*(x-m)) <= 1.
        # This is stable.
        
        shifted = all_terms - m
        exp_terms = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_terms))
        
        soft_min = m - (1.0 / k) * log_sum
        
        # We want to maximize soft_min, so minimize -soft_min
        return -soft_min

    best_centers = initial_centers.copy()
    best_score = -1.0
    
    # Annealing: start with low k (smooth), increase k (sharp)
    k_values = [10, 50, 100, 200, 500]
    
    for k in k_values:
        try:
            result = minimize(
                objective, 
                best_centers.reshape(-1), 
                args=(k,),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            # Evaluate the true radius for the result
            centers = result.x.reshape(N, 2)
            
            # Calculate true radius
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            
            wall_dists = 0.5 - np.abs(centers)
            min_wall = np.min(wall_dists)
            
            true_r = min(min_pair, min_wall)
            
            if true_r > best_score:
                best_score = true_r
                best_centers = centers
                
        except Exception:
            pass
            
    # Final refinement with very high k
    try:
        result = minimize(
            objective, 
            best_centers.reshape(-1), 
            args=(1000,),
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        centers = result.x.reshape(N, 2)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        true_r = min(min_pair, min_wall)
        
        if true_r > best_score:
            best_score = true_r
            best_centers = centers
    except Exception:
        pass

    # Ensure validity
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
