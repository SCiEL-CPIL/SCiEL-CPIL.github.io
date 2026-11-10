# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    n = 14
    d = 3
    start_time = time.time()
    timeout = 580.0  # Safety margin before 630s limit

    def objective(x):
        """
        Objective function to minimize: - (d_min / d_max)^2
        x: 1D array of length n*d
        """
        if time.time() - start_time > timeout:
            raise TimeoutError()

        points = x.reshape(n, d)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        points = points / norms
        
        dists = pdist(points)
        
        if len(dists) == 0:
            return 0.0

        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-15 or d_min < 1e-15:
            return 1e6

        ratio = d_min / d_max
        return - (ratio ** 2)

    def score(points):
        """Compute the actual score (d_min/d_max)^2"""
        points = points.reshape(n, d)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        points = points / norms
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 0.0
        return (d_min / d_max) ** 2

    # Generate good initial configurations
    def fib_sphere(n_pts):
        """Fibonacci sphere distribution"""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_pts):
            y = 1 - (i / float(n_pts - 1)) * 2
            radius = np.sqrt(max(0, 1 - y * y))
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    def random_sphere(n_pts, seed):
        """Random points on sphere"""
        rng = np.random.RandomState(seed)
        pts = rng.randn(n_pts, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        return pts

    # Spherical code initialization using known good configurations
    def spherical_code_init(n_pts):
        """Use a mix of structured and random initializations"""
        inits = []
        inits.append(fib_sphere(n_pts))
        for seed in range(10):
            inits.append(random_sphere(n_pts, seed))
        return inits

    best_points = None
    best_score = 0.0

    # Phase 1: Global optimization with differential evolution
    bounds = [(-1, 1)] * (n * d)
    
    if time.time() - start_time < timeout - 120:
        try:
            de_result = differential_evolution(
                objective, bounds, 
                seed=42, maxiter=100, tol=1e-8,
                popsize=30, mutation=(0.5, 1.5), recombination=0.7,
                polish=False, init='sobol'
            )
            if de_result.fun < 1e5:
                candidate = de_result.x.reshape(n, d)
                s = score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate
        except Exception:
            pass

    # Phase 2: Local refinement from multiple starting points
    inits = spherical_code_init(n)
    
    for init_pts in inits:
        if time.time() - start_time > timeout - 60:
            break
            
        x0 = init_pts.flatten()
        
        # Try Nelder-Mead
        try:
            result = minimize(
                objective, x0, method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000, 'maxfev': 10000}
            )
            if result.fun < 1e5:
                candidate = result.x.reshape(n, d)
                s = score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate
        except Exception:
            pass
        
        # Try Powell method
        try:
            result = minimize(
                objective, x0, method='Powell',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 3000}
            )
            if result.fun < 1e5:
                candidate = result.x.reshape(n, d)
                s = score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate
        except Exception:
            pass

    # Phase 3: Fine-tuning with very tight tolerances
    if best_points is not None and time.time() - start_time < timeout - 30:
        x0 = best_points.flatten()
        try:
            result = minimize(
                objective, x0, method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 10000, 'maxfev': 20000}
            )
            if result.fun < 1e5:
                candidate = result.x.reshape(n, d)
                s = score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate
        except Exception:
            pass

    # Final: if no good result, use Fibonacci sphere
    if best_points is None:
        best_points = fib_sphere(n)

    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = fib_sphere(n)

    return best_points

# EVOLVE-BLOCK-END
