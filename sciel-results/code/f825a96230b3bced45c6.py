# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Maximizes the min/max distance ratio for 16 points in 2D.
    
    Strategy: 
    1. Reframe the problem as maximizing a smooth surrogate of the minimum distance.
    2. Use L-BFGS-B gradient-based optimization.
    3. Normalize points such that the maximum distance is 1 (or fixed scale) to handle scale invariance.
    4. Perform multiple random restarts to escape local minima.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    def time_exceeded():
        return time.time() > deadline

    def get_score(points):
        """Calculate the actual score R = (d_min / d_max)^2."""
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    def soft_min_objective(points_flat, beta=100.0):
        """
        Objective function for minimization.
        We want to maximize the soft-min of distances, normalized by max distance.
        So we minimize -soft_min(distances) / max_distance.
        
        soft_min(d) = -1/beta * log(sum(exp(-beta * d_i)))
        This approximates min(d_i) as beta -> infinity.
        """
        points = points_flat.reshape(-1, 2)
        dists = pdist(points, 'euclidean')
        
        # Avoid log(0) or negative values
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 1e10 # Penalty for degenerate points
            
        # Normalize distances by d_max to make the problem scale-invariant
        # We want to maximize min(d)/max(d).
        # Let f = min(d)/max(d).
        # soft_min(d) approximates min(d).
        # So we approximate f by soft_min(d) / d_max.
        
        # To ensure numerical stability in exp, subtract max
        # But here we are taking exp(-beta * d). Since d > 0, this is fine.
        # However, if d is large, exp(-beta*d) underflows to 0.
        # The sum will be dominated by the smallest d.
        
        # Let's use a more stable formulation if needed, but for beta=100 and d~1, it's okay.
        # Actually, let's just maximize the soft min of the normalized distances.
        # normalized_d = dists / d_max
        # We want to maximize soft_min(normalized_d).
        
        norm_dists = dists / d_max
        
        # Soft min of norm_dists
        # To avoid underflow in exp(-beta * x) when x is small (close to 0), 
        # note that exp(-beta * x) is largest when x is smallest.
        # So sum is dominated by min.
        
        # Clip to avoid issues
        norm_dists = np.clip(norm_dists, 1e-6, 1.0)
        
        # Calculate soft min
        # Using log-sum-exp trick for stability:
        # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
        # = -1/beta * log(exp(-beta * min_x) * sum(exp(-beta * (x - min_x))))
        # = min_x - 1/beta * log(sum(exp(-beta * (x - min_x))))
        
        min_val = np.min(norm_dists)
        shifted = norm_dists - min_val
        # exp(-beta * shifted) is <= 1. 
        # sum(exp(-beta * shifted)) is roughly the number of points near the min.
        
        # For simplicity and speed, since n=16 is small, direct calculation is okay if beta isn't huge.
        # Let's use beta=50 to be safe with floating point.
        b = 50.0
        vals = np.exp(-b * shifted)
        log_sum = np.log(np.sum(vals))
        soft_min_val = min_val - (1.0 / b) * log_sum
        
        # We want to maximize soft_min_val.
        # Minimize -soft_min_val.
        return -soft_min_val

    def run_optimization(init_points):
        """
        Run L-BFGS-B optimization from a given initial configuration.
        """
        # Flatten points for optimizer
        x0 = init_points.flatten()
        
        # Constraints: None, but we rely on the objective to handle scale.
        # We don't strictly constrain d_max=1 because the optimizer might scale points.
        # However, the objective is scale-invariant (ratio), so the optimizer might drift in scale.
        # To prevent this, we can periodically normalize or add a penalty.
        # Alternatively, we can fix the scale by constraining one point or the centroid.
        # Let's just run it and normalize the result before scoring.
        
        # L-BFGS-B is good for this.
        # Max iterations to avoid hanging on a single run
        res = minimize(
            soft_min_objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        best_points = res.x.reshape(-1, 2)
        
        # Normalize to d_max = 1 for consistent scoring? 
        # The score function handles normalization internally, so it's not strictly necessary 
        # for the score, but it helps if we use this as a restart for another optimization.
        dists = pdist(best_points, 'euclidean')
        d_max = np.max(dists)
        if d_max > 1e-10:
            best_points = best_points / d_max
            
        return best_points

    # Initial configurations
    inits = []
    
    # 1. Grid-like init
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    inits.append(grid_points)
    
    # 2. Circle init
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_points = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(circle_points)
    
    # 3. Hexagonal-like init
    hex_points = []
    for row in range(-3, 4):
        for col in range(-3, 4):
            x = col + (row % 2) * 0.5
            y = row * 0.866
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    # Select 16 closest to center
    centers = np.argpartition(np.sum(hex_points**2, axis=1), 16)[:16]
    inits.append(hex_points[centers])
    
    # 4. Random inits
    for seed in range(20):
        np.random.seed(seed)
        inits.append(np.random.uniform(-1, 1, (n, 2)))

    best_points = None
    best_score = 0.0

    # Run optimization on multiple inits
    # We have ~590 seconds. 
    # Each L-BFGS-B run is relatively fast (200 iters * 120 pairs * gradient calc).
    # We can run many restarts.
    
    for i, init in enumerate(inits):
        if time_exceeded():
            break
            
        # Run optimization
        # To increase diversity, we can slightly perturb the init
        perturbed_init = init + np.random.randn(n, 2) * 0.05
        
        try:
            p = run_optimization(perturbed_init)
            s = get_score(p)
            
            if s > best_score:
                best_score = s
                best_points = p
        except Exception as e:
            # If optimization fails, skip
            pass

    # Final Polish: Take the best found and run a very high-precision optimization
    # Increase beta for sharper min approximation
    if best_points is not None and not time_exceeded():
        # We can try to refine with a higher beta or more iterations
        # For now, just run the same optimizer again from the best point found
        # This might help if the first run didn't converge fully
        try:
            # Use a slightly perturbed version to avoid exact local minimum if it's a saddle
            p_refined = run_optimization(best_points + np.random.randn(n, 2) * 0.01)
            s_refined = get_score(p_refined)
            if s_refined > best_score:
                best_score = s_refined
                best_points = p_refined
        except Exception:
            pass

    # Fallback
    if best_points is None:
        np.random.seed(0)
        best_points = np.random.randn(n, 2)
        dists = pdist(best_points, 'euclidean')
        d_max = np.max(dists)
        if d_max > 1e-10:
            best_points = best_points / d_max

    return best_points

# EVOLVE-BLOCK-END
