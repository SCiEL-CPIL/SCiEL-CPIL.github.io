import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Constructs 8 points maximizing the product of distances at fixed diameter.
    Uses a parameterization of two concentric squares.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin

    # Parameters: R1, R2, theta2
    # We fix R1 and R2 relative to each other by fixing the scale later, 
    # but it's easier to optimize R1, R2, theta2 directly and then scale.
    # Actually, since the objective is scale invariant, we can fix one radius, 
    # e.g., R1=1, and optimize R2/R1 and theta2. 
    # However, the diameter constraint couples them.
    # Let's optimize R1, R2, theta2 directly.
    
    def points_from_params(params):
        R1, R2, theta2 = params
        # Square 1: angles 0, pi/2, pi, 3pi/2
        a1 = 0.0
        p1 = [np.array([R1 * np.cos(a1 + np.pi/2 * k), R1 * np.sin(a1 + np.pi/2 * k)]) for k in range(4)]
        
        # Square 2: angles theta2, theta2+pi/2, ...
        p2 = [np.array([R2 * np.cos(theta2 + np.pi/2 * k), R2 * np.sin(theta2 + np.pi/2 * k)]) for k in range(4)]
        
        return p1 + p2

    def diameter(pts):
        # Calculate max pairwise distance
        # 8 points, 28 unique pairs
        max_d = 0.0
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(pts[i] - pts[j])
                if d > max_d:
                    max_d = d
        return max_d

    def log_score(params):
        pts = points_from_params(params)
        D = diameter(pts)
        if D == 0:
            return -np.inf
        s = 2.0 / D
        # Product of distances for ordered pairs
        # log(prod) = sum(log(dist))
        log_prod = 0.0
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                d = np.linalg.norm(pts[i] - pts[j])
                if d == 0:
                    return -np.inf
                log_prod += np.log(d)
        
        # Score S = (s^56 * prod) / 8^8
        # log(S) = 56 * log(s) + log_prod - 8 * log(8)
        log_s = np.log(s)
        log_S = 56.0 * log_s + log_prod - 8.0 * np.log(8.0)
        return log_S

    def neg_log_score(params):
        val = log_score(params)
        if not np.isfinite(val):
            return 1e10
        return -val

    # Initial guesses
    # Regular octagon corresponds to R1=R2, theta2=pi/8 (45 deg) or similar?
    # Regular octagon vertices: angles k*pi/4.
    # Square 1: 0, 90, 180, 270.
    # Square 2: 45, 135, 225, 315.
    # So R1=R2, theta2=pi/4.
    # Let's try a grid of initial points to avoid local minima.
    
    best_score = -np.inf
    best_params = None
    
    # Grid search for initialization
    # R1, R2 in [0.5, 2.0], theta2 in [0, pi/2]
    # Since scale is invariant, the ratio R2/R1 matters.
    # Let's fix R1=1 and optimize r=R2/R1 and theta.
    # But the diameter depends on absolute values? No, scale invariant.
    # So we can fix R1=1.
    
    def log_score_r_theta(r, theta):
        params = [1.0, r, theta]
        return log_score(params)
    
    def neg_log_score_r_theta(x):
        r, theta = x
        if r <= 0:
            return 1e10
        return -log_score_r_theta(r, theta)

    # Sample initial points
    init_candidates = []
    # Regular octagon
    init_candidates.append([1.0, np.pi/4])
    # Some randoms
    np.random.seed(42)
    for _ in range(50):
        r = np.random.uniform(0.2, 3.0)
        theta = np.random.uniform(0, np.pi/2)
        init_candidates.append([r, theta])
        
    # Also try extreme cases
    init_candidates.append([0.5, np.pi/4])
    init_candidates.append([2.0, np.pi/4])
    init_candidates.append([1.0, 0.0])
    init_candidates.append([1.0, np.pi/8])
    
    best_log_s = -np.inf
    best_x = None
    
    for x0 in init_candidates:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(neg_log_score_r_theta, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000, 'maxfev': 2000})
            if res.fun < best_log_s: # Note: res.fun is negative log score
                best_log_s = res.fun
                best_x = res.x
        except Exception:
            pass

    # If we found a good point, use it.
    # Also run a few more local optimizations from the best found point with different methods?
    # Nelder-Mead is robust enough for 2D.
    
    if best_x is None:
        # Fallback to regular octagon
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
        
    r_opt, theta_opt = best_x
    R1 = 1.0
    R2 = r_opt
    
    pts = points_from_params([R1, R2, theta_opt])
    
    # Scale to diameter 2
    D = diameter(pts)
    if D == 0:
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
        
    s = 2.0 / D
    scaled_pts = [p * s for p in pts]
    
    return [[float(p[0]), float(p[1])] for p in scaled_pts]

# EVOLVE-BLOCK-END
