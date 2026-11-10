# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin before 630s kill

    best_score = -1.0
    best_points = None

    def _compute_metric(points: np.ndarray) -> float:
        """Compute (d_min / d_max)^2 for the given points."""
        if points.shape[0] < 2:
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        if d_min <= 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def _fast_metric(points: np.ndarray) -> float:
        """Faster metric computation using vectorized distances."""
        n_pts = points.shape[0]
        sq_dists = np.sum(points**2, axis=1)[:, np.newaxis] + np.sum(points**2, axis=1)[np.newaxis, :] - 2 * points @ points.T
        np.fill_diagonal(sq_dists, 0.0)
        sq_dists = np.maximum(sq_dists, 0.0)
        d_min_sq = np.min(sq_dists[sq_dists > 0]) if np.any(sq_dists > 0) else 0.0
        d_max_sq = np.max(sq_dists)
        if d_max_sq <= 0:
            return 0.0
        d_min = np.sqrt(d_min_sq)
        d_max = np.sqrt(d_max_sq)
        if d_min <= 0:
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

    # Objective function for the optimizer
    # We pass raw coordinates, but we project them to the unit sphere inside.
    # This effectively optimizes the directions.
    def objective(x):
        pts = x.reshape(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts_on_sphere = pts / norms
        metric = _fast_metric(pts_on_sphere)
        return -metric

    # Simulated Annealing
    def anneal(initial_points, max_iters=100000, start_temp=0.1, end_temp=0.001):
        nonlocal best_score, best_points
        
        current_points = initial_points.copy()
        current_score = _fast_metric(current_points)
        
        if time.time() > deadline - 10:
            return current_points, current_score
            
        for i in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Generate perturbation
            noise_scale = 0.05 * (1 - i / max_iters)
            perturbed = current_points + np.random.randn(n, d) * noise_scale
            
            # Normalize to sphere
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            perturbed = perturbed / norms
            
            perturbed_score = _fast_metric(perturbed)
            
            # Acceptance criterion
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

    # Initializations
    np.random.seed(42)
    
    # 1. Icosahedron-based initialization (12 vertices) + 2 random
    phi = (1 + 5**0.5) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / ico_norms
    
    # 2. Fibonacci Sphere (uniform distribution)
    def fibonacci_sphere(n):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - (i + 0.5) * 2 / n
            radius = np.sqrt(1 - z*z)
            theta = 2 * np.pi * i / golden_ratio
            x = np.cos(theta) * radius
            y = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    fib_pts = fibonacci_sphere(n)

    # 3. Random points
    random_pts = np.random.randn(n, d)
    random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)

    # Create a list of starting points
    start_points = [
        np.vstack([ico, np.random.randn(2, d)]),
        fib_pts,
        random_pts
    ]
    
    # Add more random restarts
    for _ in range(10):
        rp = np.random.randn(n, d)
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        start_points.append(rp)

    # Optimization Loop
    for i, init_pts in enumerate(start_points):
        if time.time() > deadline - 20:
            break
            
        x0 = init_pts.flatten()
        
        try:
            # Nelder-Mead
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'maxfev': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
            candidate = res.x.reshape(n, d)
            _try_candidate(candidate)
            
            # If we have time, run Simulated Annealing from the best point found so far
            if time.time() < deadline - 30 and best_points is not None:
                # Refine the best point found so far using Simulated Annealing
                # Use a smaller number of iterations to fit within time budget
                max_sa_iters = 50000
                sa_result, sa_score = anneal(best_points, max_iters=max_sa_iters)
                _try_candidate(sa_result)
                
        except Exception:
            pass

    # Final Fallback
    if best_points is None:
        best_points = fib_pts

    # Ensure the returned points are finite and have the correct shape
    best_points = best_points.reshape(n, d)
    if not np.all(np.isfinite(best_points)):
        best_points = fib_pts

    return best_points

# EVOLVE-BLOCK-END
