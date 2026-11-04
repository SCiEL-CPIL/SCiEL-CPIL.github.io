"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 1.0

def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_tri = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return min_tri / hull_area

def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin with given radius."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    """Construct 14 points to maximize the minimum triangle area over convex hull area."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with regular 14-gon
    points = _regular_polygon(N, radius=1.0)
    best_score = _score(points)
    best_points = points.copy()
    
    # Try to improve using differential evolution in a Fourier-based parameterization
    # Represent points as radial perturbations of regular polygon
    # x_i = (1 + a_k * cos(k*theta_i) + b_k * sin(k*theta_i)) * (cos(theta_i), sin(theta_i))
    # This keeps points roughly on the unit circle but allows distortion
    
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    def decode(params):
        """Decode Fourier coefficients to 2D points."""
        # params: [a1, b1, a2, b2, ..., aK, bK]
        K = len(params) // 2
        r = np.ones(N)
        for k in range(1, K + 1):
            a = params[2 * (k - 1)]
            b = params[2 * (k - 1) + 1]
            r += a * np.cos(k * theta) + b * np.sin(k * theta)
        # Ensure positive radii
        r = np.maximum(r, 0.1)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    def objective(params):
        pts = decode(params)
        return -_score(pts)
    
    # Use a few Fourier modes
    n_modes = 4
    n_params = 2 * n_modes
    bounds = [(-0.3, 0.3)] * n_params
    
    # Run differential evolution
    de_result = differential_evolution(
        objective, bounds, seed=42, maxiter=50, popsize=15,
        tol=1e-6, polish=False, init='sobol'
    )
    
    if de_result.fun < -best_score:
        best_points = decode(de_result.x)
        best_score = -de_result.fun
    
    # Polish with L-BFGS-B
    if time.time() < deadline:
        res = minimize(
            objective, de_result.x, method='L-BFGS-B',
            bounds=bounds, options={'maxiter': 200}
        )
        if res.fun < -best_score:
            best_points = decode(res.x)
            best_score = -res.fun
    
    # Final polish with Powell
    if time.time() < deadline:
        res = minimize(
            objective, best_points.reshape(-1) if False else res.x if hasattr(res, 'x') else de_result.x,
            method='Powell', options={'maxiter': 100}
        )
        if res.fun < -best_score:
            best_points = decode(res.x)
            best_score = -res.fun
    
    return best_points
# EVOLVE-BLOCK-END
