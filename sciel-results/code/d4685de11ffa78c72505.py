import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Maximize (d_min / d_max)^2 for 16 points in 2D.
    
    Strategy:
    1. Define a smooth surrogate objective using soft-min and soft-max of pairwise distances.
       This makes the objective differentiable, allowing gradient-based optimization (L-BFGS-B)
       to navigate the non-convex landscape more effectively than derivative-free methods
       on the non-smooth original objective.
    2. Initialize with diverse configurations (grid, rings, random) to avoid local minima.
    3. Optimize using L-BFGS-B with the smooth surrogate.
    4. Evaluate the true objective on the result and keep the best.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    def compute_true_score(points):
        """Compute the actual metric R = (d_min / d_max)^2."""
        if points.shape != (16, 2):
            return 0.0
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def soft_min_max_score(points, t=100.0):
        """
        Smooth surrogate for (d_min / d_max)^2.
        Uses soft-min for d_min and soft-max for d_max.
        soft_min(d, t) = -1/t * log(sum(exp(-t*d)))
        soft_max(d, t) = 1/t * log(sum(exp(t*d)))
        To avoid overflow, we shift d by its mean.
        """
        d = pdist(points)
        if d.size == 0:
            return -1e10
        
        # Shift to prevent overflow in exp
        d_shifted = d - np.mean(d)
        
        # Soft min: approximate min(d)
        # min(d) ≈ -1/t * log(sum(exp(-t*(d - mean))))
        exp_neg = np.exp(-t * d_shifted)
        log_sum_exp_neg = np.log(np.sum(exp_neg))
        soft_min_val = -log_sum_exp_neg / t - np.mean(d)
        
        # Soft max: approximate max(d)
        # max(d) ≈ 1/t * log(sum(exp(t*(d - mean))))
        exp_pos = np.exp(t * d_shifted)
        log_sum_exp_pos = np.log(np.sum(exp_pos))
        soft_max_val = log_sum_exp_pos / t + np.mean(d)
        
        if soft_max_val < 1e-12:
            return -1e10
        
        ratio = soft_min_val / soft_max_val
        # We want to maximize ratio^2, so minimize -ratio^2
        return -ratio * ratio

    def objective_wrapper(x):
        """Wrapper for scipy.optimize.minimize, expects 1D array of length 32."""
        pts = x.reshape(16, 2)
        return soft_min_max_score(pts, t=100.0)

    def gradient_wrapper(x):
        """
        Numerical gradient approximation for the smooth surrogate.
        Since deriving the analytic gradient of the soft-min/max of pdist is complex,
        we rely on L-BFGS-B's finite difference gradient or provide a simple numerical one.
        Actually, L-BFGS-B can do finite differences, but it's slow.
        Let's try to provide a numerical gradient explicitly or just let scipy do it.
        For speed, let's use a central difference with a small step.
        """
        eps = 1e-5
        g = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            g[i] = (objective_wrapper(x_plus) - objective_wrapper(x_minus)) / (2 * eps)
        return g

    # Initial configurations
    def make_grid_init():
        coords = np.array([[-1.5, -0.5, 0.5, 1.5] * 4,
                           [-1.5, -1.5, -1.5, -1.5,
                            -0.5, -0.5, -0.5, -0.5,
                             0.5,  0.5,  0.5,  0.5,
                             1.5,  1.5,  1.5,  1.5]]).T
        # Normalize to unit radius
        max_r = np.max(np.linalg.norm(coords, axis=1))
        return coords / max_r

    def make_ring_init(radii, angles_per_ring):
        pts = []
        for r, n in zip(radii, angles_per_ring):
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            for a in angles:
                pts.append([r * np.cos(a), r * np.sin(a)])
        return np.array(pts)

    inits = []
    
    # 1. Grid
    inits.append(make_grid_init())
    
    # 2. Two rings: 4 and 12
    inits.append(make_ring_init([0.5, 1.0], [4, 12]))
    
    # 3. Three rings: 4, 8, 4
    inits.append(make_ring_init([0.3, 0.7, 1.0], [4, 8, 4]))
    
    # 4. Random (seeded for reproducibility)
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.randn(16, 2)
        max_r = np.max(np.linalg.norm(pts, axis=1))
        inits.append(pts / max_r)

    best_points = None
    best_score = -np.inf

    # Evaluate initial scores
    for init in inits:
        score = compute_true_score(init)
        if score > best_score:
            best_score = score
            best_points = init.copy()

    # Optimize from each initialization
    for i, init in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        x0 = init.flatten()
        
        try:
            res = minimize(
                objective_wrapper, 
                x0, 
                jac=gradient_wrapper, # Providing gradient helps L-BFGS-B
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
            )
            
            opt_pts = res.x.reshape(16, 2)
            opt_score = compute_true_score(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
                
        except Exception:
            continue

    # If time remains, do a few more random restarts or refine the best point
    if best_points is not None and time.time() < deadline - 60:
        # Refine the best point found so far with a tighter tolerance
        x0 = best_points.flatten()
        try:
            res = minimize(
                objective_wrapper, 
                x0, 
                jac=gradient_wrapper,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
            )
            opt_pts = res.x.reshape(16, 2)
            opt_score = compute_true_score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = make_grid_init()

    return best_points
