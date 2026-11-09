import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # safety margin for 630s kill

    def score(pts):
        """Compute (d_min / d_max)^2 for the point configuration."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d) or not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def fibonacci_sphere_init():
        """Generate initial points on a Fibonacci sphere."""
        indices = np.arange(0, n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        pts = np.column_stack([x, y, z])
        pts = pts - np.mean(pts, axis=0)
        return pts

    def make_penalty_objective(beta):
        """
        Create an objective that penalizes variance in pairwise distances.
        We want all distances to be as equal as possible.
        Objective: mean(d^2) - beta * var(d^2) / mean(d^2)
        This is scale invariant and smooth.
        """
        def objective(x):
            pts = x.reshape((n, d))
            pts = pts - np.mean(pts, axis=0)
            dists_sq = pdist(pts) ** 2
            mean_d2 = np.mean(dists_sq)
            if mean_d2 < 1e-15:
                return 0.0
            var_d2 = np.var(dists_sq)
            # Minimize variance relative to mean
            return var_d2 / mean_d2
        
        def gradient(x):
            pts = x.reshape((n, d))
            pts = pts - np.mean(pts, axis=0)
            dists_sq = pdist(pts) ** 2
            mean_d2 = np.mean(dists_sq)
            if mean_d2 < 1e-15:
                return np.zeros_like(x)
            var_d2 = np.var(dists_sq)
            
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                f_plus = objective(x_plus)
                f_minus = objective(x_minus)
                grad[i] = (f_plus - f_minus) / (2 * eps)
            return grad
        
        return objective, gradient

    best_score = 0.0
    best_points = None

    # Multiple initializations
    inits = []
    inits.append(fibonacci_sphere_init())
    
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = rng.randn(n, d)
        pts = pts - np.mean(pts, axis=0)
        # Normalize to unit scale
        dists = pdist(pts)
        if len(dists) > 0 and np.max(dists) > 1e-12:
            pts = pts / np.max(dists)
        inits.append(pts)

    # Optimize from each initialization with increasing penalty
    betas = [1.0, 10.0, 100.0, 1000.0]
    
    for init_pts in inits:
        if time.time() > deadline - 60:
            break
        
        current_pts = init_pts.copy()
        
        for beta in betas:
            if time.time() > deadline - 30:
                break
            
            try:
                obj, grad = make_penalty_objective(beta)
                x0 = current_pts.flatten()
                
                res = minimize(
                    obj,
                    x0,
                    jac=grad,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                
                if res.x is not None:
                    cand = res.x.reshape((n, d))
                    cand = cand - np.mean(cand, axis=0)
                    s = score(cand)
                    if s > best_score:
                        best_score = s
                        best_points = cand.copy()
                        current_pts = cand.copy()
            except Exception:
                pass

    # Final local refinement with SLSQP on the actual ratio objective
    if best_points is not None and time.time() < deadline - 30:
        def ratio_objective(x):
            pts = x.reshape((n, d))
            pts = pts - np.mean(pts, axis=0)
            dists = pdist(pts)
            if len(dists) == 0:
                return 0.0
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-12:
                return 0.0
            return -(d_min / d_max) ** 2  # Negate for minimization
        
        def ratio_gradient(x):
            eps = 1e-7
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                f_plus = ratio_objective(x_plus)
                f_minus = ratio_objective(x_minus)
                grad[i] = (f_plus - f_minus) / (2 * eps)
            return grad
        
        try:
            x0 = best_points.flatten()
            res = minimize(
                ratio_objective,
                x0,
                jac=ratio_gradient,
                method='SLSQP',
                options={'maxiter': 2000, 'ftol': 1e-15}
            )
            if res.x is not None:
                cand = res.x.reshape((n, d))
                cand = cand - np.mean(cand, axis=0)
                s = score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
        except Exception:
            pass

    # Final normalization
    if best_points is not None and np.all(np.isfinite(best_points)):
        dists = pdist(best_points)
        d_max = np.max(dists)
        if d_max > 1e-12:
            best_points = best_points / d_max
        best_points = best_points - np.mean(best_points, axis=0)
    else:
        best_points = fibonacci_sphere_init()

    return best_points
