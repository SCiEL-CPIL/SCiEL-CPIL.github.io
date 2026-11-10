# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin before 630s kill

    n = 16
    d = 2

    def compute_score(points):
        """Compute R = (d_min / d_max)^2"""
        if len(points) != 16 or len(points[0]) != 2:
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(params, softness=0.1):
        """
        Smooth surrogate: maximize min distance / max distance.
        Use log-sum-exp for soft max, and -log-sum-exp for soft min.
        Objective to minimize: -log(d_min_soft) + log(d_max_soft)
        """
        points = params.reshape(-1, 2)
        dists = pdist(points)
        if np.any(dists < 1e-15):
            return 1e6  # Penalize coincident points
        
        # Soft max: log(sum(exp(dists/softness))) * softness approximates max
        d_max_soft = softness * np.log(np.sum(np.exp(dists / softness)))
        # Soft min: -log(sum(exp(-dists/softness))) * softness approximates min
        d_min_soft = -softness * np.log(np.sum(np.exp(-dists / softness)))
        
        if d_max_soft < 1e-10:
            return 1e6
        return -np.log(d_min_soft) + np.log(d_max_soft)

    def objective(params):
        return smooth_objective(params, softness=0.05)

    def gradient(params):
        """Numerical gradient"""
        eps = 1e-7
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            grad[i] = (objective(params_plus) - objective(params_minus)) / (2 * eps)
        return grad

    # Generate diverse initial points
    def random_points(seed):
        np.random.seed(seed)
        points = np.random.randn(n, d)
        # Center
        points -= np.mean(points, axis=0)
        # Scale so that sum of squares = n
        scale = np.sqrt(np.sum(points**2) / n)
        if scale > 1e-10:
            points /= scale
        return points

    def grid_points():
        """4x4 grid centered at origin"""
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack((xx.ravel(), yy.ravel()))
        points -= np.mean(points, axis=0)
        scale = np.sqrt(np.sum(points**2) / n)
        if scale > 1e-10:
            points /= scale
        return points

    def hexagonal_points():
        """Approximate hexagonal packing"""
        points = []
        # Center point
        points.append([0.0, 0.0])
        # Ring 1: 6 points
        for k in range(6):
            angle = k * np.pi / 3
            points.append([np.cos(angle), np.sin(angle)])
        # Ring 2: 12 points (take 9 to get 16 total)
        for k in range(12):
            angle = k * np.pi / 6 + np.pi / 6
            r = 2.0
            points.append([r * np.cos(angle), r * np.sin(angle)])
        points = np.array(points[:16])
        points -= np.mean(points, axis=0)
        scale = np.sqrt(np.sum(points**2) / n)
        if scale > 1e-10:
            points /= scale
        return points

    def circle_points():
        """16 points on a circle"""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        return points

    # Initial candidates
    initial_points = [
        random_points(42),
        random_points(123),
        random_points(456),
        random_points(789),
        random_points(321),
        grid_points(),
        hexagonal_points(),
        circle_points(),
    ]

    best_score = -np.inf
    best_points = initial_points[0].copy()

    # Evaluate initial points
    for i, pts in enumerate(initial_points):
        score = compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Try optimization from multiple starting points
    for i, init_pts in enumerate(initial_points):
        if time.time() > deadline - 60:
            break

        params = init_pts.flatten()
        
        # Try with different softness values
        for softness in [0.05, 0.1, 0.2]:
            if time.time() > deadline - 60:
                break

            def obj(params):
                points = params.reshape(-1, 2)
                dists = pdist(points)
                if np.any(dists < 1e-15):
                    return 1e6
                d_max_soft = softness * np.log(np.sum(np.exp(dists / softness)))
                d_min_soft = -softness * np.log(np.sum(np.exp(-dists / softness)))
                if d_max_soft < 1e-10:
                    return 1e6
                return -np.log(d_min_soft) + np.log(d_max_soft)

            try:
                result = minimize(obj, params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if result.success or result.fun < 1e5:
                    opt_points = result.x.reshape(-1, 2)
                    score = compute_score(opt_points)
                    if score > best_score:
                        best_score = score
                        best_points = opt_points.copy()
            except Exception:
                pass

    # Final polish: local search with very small perturbations
    if time.time() < deadline - 30:
        current = best_points.copy()
        current_score = best_score
        for _ in range(2000):
            if time.time() > deadline - 10:
                break
            # Small random perturbation
            perturbation = np.random.randn(n, d) * 0.01
            new_points = current + perturbation
            new_score = compute_score(new_points)
            if new_score > current_score:
                current = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current.copy()
            else:
                # Try smaller perturbation
                perturbation = np.random.randn(n, d) * 0.001
                new_points = current + perturbation
                new_score = compute_score(new_points)
                if new_score > current_score:
                    current = new_points
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current.copy()

    return best_points

# EVOLVE-BLOCK-END
