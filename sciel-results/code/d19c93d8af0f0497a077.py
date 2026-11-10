# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses Differential Evolution for global search and Nelder-Mead for local refinement.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 615.0  # Safety margin

    best_score = -1.0
    best_points = None

    def _fast_metric(points: np.ndarray) -> float:
        """Compute (d_min / d_max)^2 for the given points using vectorized ops."""
        if points.shape[0] < 2:
            return 0.0
        # Vectorized squared distances
        sq_dists = np.sum(points**2, axis=1)[:, np.newaxis] + np.sum(points**2, axis=1)[np.newaxis, :] - 2 * points @ points.T
        np.fill_diagonal(sq_dists, 0.0)
        sq_dists = np.maximum(sq_dists, 0.0)
        
        # Get min and max squared distances
        # Exclude diagonal (0)
        mask = sq_dists > 1e-10
        if not np.any(mask):
            return 0.0
            
        d_min_sq = np.min(sq_dists[mask])
        d_max_sq = np.max(sq_dists)
        
        if d_max_sq <= 1e-10:
            return 0.0
            
        d_min = np.sqrt(d_min_sq)
        d_max = np.sqrt(d_max_sq)
        
        if d_min <= 1e-10:
            return 0.0
            
        ratio = d_min / d_max
        return ratio * ratio

    def _try_candidate(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            return
        if not np.all(np.isfinite(pts)):
            return
        score = _fast_metric(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def objective(x):
        pts = x.reshape(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts_on_sphere = pts / norms
        metric = _fast_metric(pts_on_sphere)
        return -metric

    # Fallback initialization: Fibonacci Sphere
    def fibonacci_sphere(n_pts):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n_pts):
            z = 1 - (i + 0.5) * 2 / n_pts
            radius = np.sqrt(1 - z*z)
            theta = 2 * np.pi * i / golden_ratio
            x = np.cos(theta) * radius
            y = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    np.random.seed(42)
    
    # 1. Differential Evolution for Global Search
    # Bounds: [-1, 1] for each coordinate. Normalization happens inside objective.
    bounds = [(-1.0, 1.0)] * (n * d)
    
    de_result = None
    try:
        if time.time() < deadline - 100:
            # DE parameters: popsize=15 is standard, maxiter=200 to keep time controlled
            # tol=1e-8
            de_result = differential_evolution(
                objective, 
                bounds, 
                maxiter=200, 
                popsize=20, 
                tol=1e-8, 
                seed=42, 
                polish=False,
                init='sobol' # Better spread than 'latinhypercube' for high dims? Or 'halton'
            )
            if de_result.success or de_result.fun < 0:
                candidate = de_result.x.reshape(n, d)
                _try_candidate(candidate)
    except Exception:
        pass

    # 2. Local Refinement with Nelder-Mead
    # Start from the best point found so far (DE result or Fibonacci)
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    x0 = best_points.flatten()
    
    try:
        if time.time() < deadline - 50:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'maxfev': 20000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            candidate = res.x.reshape(n, d)
            _try_candidate(candidate)
    except Exception:
        pass

    # 3. Simulated Annealing for final polish / escape
    def anneal(initial_points, max_iters=20000, start_temp=0.05, end_temp=0.001):
        nonlocal best_score, best_points
        
        current_points = initial_points.copy()
        current_score = _fast_metric(current_points)
        
        if time.time() > deadline - 10:
            return current_points, current_score
            
        for i in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            noise_scale = 0.02 * (1 - i / max_iters)
            perturbed = current_points + np.random.randn(n, d) * noise_scale
            
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            perturbed = perturbed / norms
            
            perturbed_score = _fast_metric(perturbed)
            
            if perturbed_score > current_score:
                current_points = perturbed
                current_score = perturbed_score
            else:
                temp = start_temp * (end_temp / start_temp) ** (i / max_iters)
                if np.random.rand() < np.exp((perturbed_score - current_score) / temp):
                    current_points = perturbed
                    current_score = perturbed_score
            
            _try_candidate(current_points)
            
        return current_points, current_score

    if best_points is not None and time.time() < deadline - 20:
        sa_result, _ = anneal(best_points)
        _try_candidate(sa_result)

    # Final Fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)

    best_points = best_points.reshape(n, d)
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)

    return best_points

# EVOLVE-BLOCK-END
