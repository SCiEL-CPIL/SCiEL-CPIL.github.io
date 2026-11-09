# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses complex-number representation with first point fixed at origin,
    log-sum-exp surrogate for minimum distance, and L-BFGS-B optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16  # total number of points
    
    def compute_score(points):
        """Compute R = (d_min / d_max)^2"""
        d = pdist(points)
        if len(d) == 0:
            return 0.0
        d_min_val = np.min(d)
        d_max_val = np.max(d)
        if d_max_val <= 0:
            return 0.0
        return (d_min_val / d_max_val) ** 2
    
    def objective(x, t=50.0):
        """
        Objective function for L-BFGS-B minimization.
        x is a 2D array of shape (15, 2) representing 15 points (z_0 = 0 fixed).
        We minimize -log(soft_min) + penalty for max distance > 1.
        """
        # x is flattened to 30 elements: [x1, y1, x2, y2, ..., x15, y15]
        pts = np.zeros((n, 2))
        pts[1:, 0] = x[0::2]
        pts[1:, 1] = x[1::2]
        
        d = pdist(pts)
        if len(d) == 0:
            return 0.0
        
        # Log-sum-exp based soft minimum
        d_min_val = np.min(d)
        # Use stable log-sum-exp: log(sum(exp(d_i - d_min))) + d_min
        shifted = d - d_min_val
        max_shifted = np.max(shifted)
        log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        soft_min = d_min_val - (1.0 / t) * log_sum_exp
        
        # Penalty for exceeding unit max distance
        d_max_val = np.max(d)
        penalty = 1000.0 * max(0.0, d_max_val - 1.0) ** 2
        
        # Maximize soft_min -> minimize -soft_min + penalty
        return -soft_min + penalty
    
    def gradient(x, t=50.0):
        """
        Numerical gradient for the objective function.
        """
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = objective(x, t)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            fp = objective(xp, t)
            xm = x.copy()
            xm[i] -= eps
            fm = objective(xm, t)
            grad[i] = (fp - fm) / (2 * eps)
        return grad
    
    best_score = -np.inf
    best_points = None
    
    # Multiple restart strategies
    n_trials = 30
    
    for trial in range(n_trials):
        np.random.seed(12345 + trial * 137)
        
        if trial < 8:
            # Strategy 1: Random points in unit disk
            r = np.sqrt(np.random.rand(15))
            theta = 2 * np.pi * np.random.rand(15)
            x0 = np.zeros(30)
            x0[0::2] = r * np.cos(theta)
            x0[1::2] = r * np.sin(theta)
        elif trial < 16:
            # Strategy 2: Structured concentric circles
            x0 = np.zeros(30)
            # 5 inner, 5 middle, 5 outer
            for i in range(15):
                if i < 5:
                    r = 0.45
                    theta = 2 * np.pi * i / 5
                elif i < 10:
                    r = 0.75
                    theta = 2 * np.pi * (i - 5) / 5 + np.pi / 5
                else:
                    r = 1.0
                    theta = 2 * np.pi * (i - 10) / 5 + np.pi / 10
                x0[2*i] = r * np.cos(theta)
                x0[2*i + 1] = r * np.sin(theta)
        elif trial < 24:
            # Strategy 3: Hexagonal lattice based
            points_hex = []
            for row in range(-2, 3):
                for col in range(-2, 3):
                    x = col + 0.5 * (row % 2)
                    y = row * np.sqrt(3) / 2
                    points_hex.append((x, y))
            
            points_hex = np.array(points_hex)
            points_hex -= np.mean(points_hex, axis=0)
            max_r = np.max(np.sqrt(points_hex[:, 0]**2 + points_hex[:, 1]**2))
            if max_r > 0:
                points_hex = points_hex / max_r * 0.95
            
            # Select 15 points closest to center
            dists_to_center = np.sqrt(points_hex[:, 0]**2 + points_hex[:, 1]**2)
            idx = np.argsort(dists_to_center)[:15]
            selected = points_hex[idx]
            
            # Add small random perturbation
            noise = 0.05 * np.random.randn(15, 2)
            selected = selected + noise
            
            x0 = np.zeros(30)
            x0[0::2] = selected[:, 0]
            x0[1::2] = selected[:, 1]
        else:
            # Strategy 4: Fibonacci sphere-like distribution projected to 2D
            x0 = np.zeros(30)
            for i in range(15):
                # Use golden ratio for angular distribution
                angle = i * 2.399963  # golden angle in radians
                r = 0.5 + 0.5 * (i / 14.0)
                x0[2*i] = r * np.cos(angle)
                x0[2*i + 1] = r * np.sin(angle)
        
        # Optimize with L-BFGS-B
        try:
            res = minimize(
                objective, x0, method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
        except Exception:
            continue
        
        # Reconstruct points
        z = np.zeros((n, 2))
        z[1:, 0] = res.x[0::2]
        z[1:, 1] = res.x[1::2]
        
        score = compute_score(z)
        
        if score > best_score:
            best_score = score
            best_points = z.copy()
    
    # Refinement: multiple passes with smaller steps around best solution
    if best_points is not None:
        for refine_pass in range(3):
            noise_scale = 0.02 / (refine_pass + 1)
            for _ in range(5):
                noise = noise_scale * np.random.randn(30)
                x_perturbed = np.zeros(30)
                x_perturbed[0::2] = best_points[1:, 0] + noise[0::2]
                x_perturbed[1::2] = best_points[1:, 1] + noise[1::2]
                
                try:
                    res = minimize(
                        objective, x_perturbed, method='L-BFGS-B',
                        options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12}
                    )
                except Exception:
                    continue
                
                z = np.zeros((n, 2))
                z[1:, 0] = res.x[0::2]
                z[1:, 1] = res.x[1::2]
                
                score = compute_score(z)
                
                if score > best_score:
                    best_score = score
                    best_points = z.copy()
    
    # Final rescaling: normalize so that d_max = 1
    if best_points is not None:
        d = pdist(best_points)
        d_max_val = np.max(d)
        if d_max_val > 0:
            best_points = best_points / d_max_val
    
    return best_points


# EVOLVE-BLOCK-END
