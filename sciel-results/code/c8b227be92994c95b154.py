import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    timeout = 590  # 10s margin
    
    def objective(flat_points):
        if time.time() - start_time > timeout:
            return np.inf
        points = flat_points.reshape(n, d)
        # Penalty for not being on unit sphere
        norms_sq = np.sum(points**2, axis=1)
        sphere_penalty = np.sum((norms_sq - 1.0)**2)
        
        # Pairwise distances
        dists = pdist(points)
        
        # Soft min and max
        # Use a beta that is large enough to approximate min/max but small enough for stability
        # beta = 100 is usually good for this scale
        beta = 50.0
        
        # Clip distances to avoid overflow in exp
        # Max possible distance on unit sphere is 2.
        # exp(50*2) is huge, but manageable in float64? 
        # exp(100) ~ 1e43. exp(50*2)=exp(100). OK.
        # But if points are far apart (not on sphere), dists can be large.
        # The penalty will push them back.
        
        # To be safe, let's use a stable log-sum-exp implementation
        def log_sum_exp(x, beta):
            # x is array of distances
            # We want (1/beta) * log(sum(exp(beta * x)))
            # This is equivalent to log(sum(exp(x))) / beta
            # Let's just compute directly with clipping
            x_scaled = beta * x
            max_x = np.max(x_scaled)
            if max_x > 700: # exp(700) overflows
                # Approximation: max_x + log(sum(exp(x_scaled - max_x)))
                # If max_x is huge, the term dominates
                return max_x / beta
            return (max_x + np.log(np.sum(np.exp(x_scaled - max_x)))) / beta
            
        def neg_log_sum_exp(x, beta):
            # -1/beta * log(sum(exp(-beta * x)))
            # = -1/beta * log(sum(exp(-beta * x)))
            x_scaled = -beta * x
            max_x = np.max(x_scaled)
            if max_x > 700:
                return max_x / beta # This would be negative? 
                # Wait. 
                # S_min = -1/beta log(sum(exp(-beta d)))
                # Let y = -beta d. S_min = -1/beta log(sum(exp(y)))
                # If y is large positive (d is large negative? No d>0), 
                # Actually d >= 0. So -beta d <= 0.
                # max_y = -beta * min(d).
                # log_sum_exp(y) = max_y + log(sum(exp(y-max_y)))
                # S_min = -1/beta * (max_y + log(sum(exp(y-max_y))))
                # = min(d) - 1/beta log(sum(exp(-beta(d - min(d)))))
                # This is stable.
                pass
            
            # Let's implement stable S_min directly
            d_min = np.min(x)
            y = -beta * (x - d_min)
            # y <= 0. max y = 0.
            # sum(exp(y)) is between 1 and N(N-1)/2
            # log(sum(exp(y))) is small.
            s_min = d_min - (1.0/beta) * np.log(np.sum(np.exp(y)))
            return s_min

        # Calculate S_max
        # S_max = 1/beta log(sum(exp(beta d)))
        # Let d_max = max(d).
        # y = beta * (d - d_max) <= 0.
        d_max = np.max(dists)
        y_max = beta * (dists - d_max)
        s_max = d_max + (1.0/beta) * np.log(np.sum(np.exp(y_max)))
        
        # Calculate S_min
        d_min = np.min(dists)
        y_min = -beta * (dists - d_min)
        s_min = d_min - (1.0/beta) * np.log(np.sum(np.exp(y_min)))
        
        # Objective: Minimize log(S_max) - log(S_min)
        # This is equivalent to minimizing log(S_max/S_min)
        # Which is equivalent to minimizing S_max/S_min
        # Which is equivalent to maximizing S_min/S_max
        
        # Avoid log(0) or negative values
        if s_min <= 0 or s_max <= 0:
            return 1e10 + sphere_penalty
            
        obj = np.log(s_max) - np.log(s_min)
        
        return obj + 100.0 * sphere_penalty

    best_points = None
    best_score = -np.inf
    
    # Generate initial guesses
    # 1. Fibonacci Sphere
    phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * np.arange(n)
    p_fib = np.column_stack((
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ))
    
    # 2. Icosahedron + 2 points (approx)
    # Icosahedron vertices
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Add 2 points at poles? Or random?
    # Let's just use random perturbations of the icosahedron
    # And random points
    
    initial_guesses = [p_fib.flatten()]
    
    # Add some random perturbations of the Fibonacci sphere
    rng = np.random.default_rng(42)
    for _ in range(5):
        noise = rng.normal(0, 0.1, (n, d))
        p_pert = p_fib + noise
        # Project to sphere
        p_pert = p_pert / np.linalg.norm(p_pert, axis=1, keepdims=True)
        initial_guesses.append(p_pert.flatten())
        
    # Add random points on sphere
    for _ in range(5):
        v = rng.normal(0, 1, (n, d))
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        initial_guesses.append(v.flatten())

    # Optimize each guess
    for x0 in initial_guesses:
        if time.time() - start_time > timeout - 10:
            break
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success or res.fun < 1e6:
                pts = res.x.reshape(n, d)
                # Normalize to unit sphere to remove any residual penalty drift
                pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
                
                # Calculate actual score
                dists = pdist(pts)
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_max > 0:
                    score = (d_min / d_max)**2
                else:
                    score = 0
                    
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            continue
            
    if best_points is None:
        # Fallback to Fibonacci sphere
        return p_fib
        
    return best_points
