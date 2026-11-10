import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize (d_min/d_max)^2.
    
    Strategy:
    1. Generate a high-quality initial configuration using a Hexagonal Lattice.
       The 16 points closest to the origin in a hexagonal lattice form a dense,
       symmetric cluster that is a very strong candidate for the global optimum.
    2. Perform a rapid, robust local optimization using Nelder-Mead on the raw
       objective function. This is much faster than gradient-based methods on
       non-smooth functions and less prone to getting stuck in local minima
       of a smooth surrogate if the starting point is already very good.
    3. Use a strict time budget to ensure completion well within 600s.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit

    def time_exceeded():
        return time.time() > deadline

    def evaluate(points):
        """Calculate the actual score R = (d_min/d_max)^2"""
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_raw(params):
        """Objective function for optimization: minimize -R"""
        points = params.reshape((n, 2))
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        ratio = d_min / d_max
        return -(ratio ** 2)

    # Step 1: Generate Hexagonal Lattice Prior
    # Hexagonal lattice points are generated using axial coordinates (q, r)
    # x = s * (q + r/2)
    # y = s * (sqrt(3)/2 * r)
    # We generate a large enough grid to ensure we have at least 16 points.
    
    def generate_hex_lattice_points(max_radius=5, spacing=1.0):
        points = []
        for q in range(-max_radius, max_radius + 1):
            for r in range(-max_radius, max_radius + 1):
                # Check if (q, r) is within a hexagonal shape of radius max_radius
                # A point (q, r) is in the hexagon of radius N if max(|q|, |r|, |q+r|) <= N
                # Actually, the standard hexagonal distance is (|q| + |r| + |q+r|) / 2
                # Let's just generate all points in a square bounding box and filter by distance later.
                x = spacing * (q + 0.5 * r)
                y = spacing * (np.sqrt(3) / 2 * r)
                points.append([x, y])
        return np.array(points)

    hex_points = generate_hex_lattice_points(max_radius=5)
    
    # Center the points
    hex_points -= np.mean(hex_points, axis=0)
    
    # Calculate distance from center for each point
    dists_to_center = np.linalg.norm(hex_points, axis=1)
    
    # Sort by distance to center and take the first 16
    sorted_indices = np.argsort(dists_to_center)
    initial_points = hex_points[sorted_indices[:n]].copy()
    
    # Normalize the initial points to have unit diameter or similar scale
    # This helps the optimizer work in a reasonable range
    d_max_init = np.max(pdist(initial_points))
    if d_max_init > 1e-10:
        initial_points = initial_points / d_max_init

    best_points = initial_points
    best_score = evaluate(best_points)

    # Step 2: Local Optimization
    # Use Nelder-Mead on the raw objective. 
    # We will run a few iterations with different initial simplex sizes to be robust.
    
    if not time_exceeded():
        # Run Nelder-Mead with adaptive settings
        try:
            res = minimize(
                objective_raw,
                initial_points.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                },
                callback=lambda xk: time_exceeded()
            )
            if not time_exceeded():
                candidate = res.x.reshape((n, 2))
                score = evaluate(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
        except Exception:
            pass

    # Step 3: If time remains, try a second round of optimization from the best point
    # with a slightly different simplex to escape potential local minima.
    if not time_exceeded():
        try:
            # Add a tiny bit of noise to the best point to perturb the simplex
            np.random.seed(42)
            perturbed = best_points + np.random.normal(0, 1e-4, (n, 2))
            res = minimize(
                objective_raw,
                perturbed.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                },
                callback=lambda xk: time_exceeded()
            )
            if not time_exceeded():
                candidate = res.x.reshape((n, 2))
                score = evaluate(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
        except Exception:
            pass

    # Final check
    if best_points is None:
        # Fallback to the initial hexagonal configuration
        best_points = initial_points

    return best_points
