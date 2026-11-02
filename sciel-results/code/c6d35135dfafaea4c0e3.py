"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import differential_evolution


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: if points are collinear or degenerate, return 0
        return 0.0


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _fourier_convex_boundary(theta, coeffs):
    """
    Generate points on a convex boundary using Fourier series.
    theta: array of angles
    coeffs: array of Fourier coefficients [a0, a1, b1, a2, b2, ...]
    Returns points in 2D.
    """
    n_theta = len(theta)
    x = np.zeros(n_theta)
    y = np.zeros(n_theta)
    
    # a0 is the mean radius
    a0 = coeffs[0] if len(coeffs) > 0 else 1.0
    
    for i in range(n_theta):
        t = theta[i]
        r = a0
        # Add Fourier terms
        for k in range(1, len(coeffs) // 2 + 1):
            a_k = coeffs[2 * k - 1] if 2 * k - 1 < len(coeffs) else 0.0
            b_k = coeffs[2 * k] if 2 * k < len(coeffs) else 0.0
            r += a_k * np.cos(k * t) + b_k * np.sin(k * t)
        
        # Ensure positive radius for convexity
        r = max(r, 0.1)
        x[i] = r * np.cos(t)
        y[i] = r * np.sin(t)
    
    return np.column_stack([x, y])


def _objective_fourier(params, n_points=N, n_fourier_terms=4):
    """
    Objective function for Fourier-based optimization.
    params: [a0, a1, b1, a2, b2, ..., a_n, b_n, phase_shift]
    We want to maximize the minimum triangle area / hull area,
    so we minimize the negative of that.
    """
    # Parse parameters
    a0 = params[0]
    n_coeffs = 2 * n_fourier_terms
    coeffs = np.array([a0] + list(params[1:1 + n_coeffs]))
    
    # Generate points on the Fourier boundary
    # We need n_points evenly spaced angles
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    # Add phase shift if provided
    if len(params) > 1 + n_coeffs:
        phase = params[1 + n_coeffs]
        theta = theta + phase
    
    points = _fourier_convex_boundary(theta, coeffs)
    
    # Check for degenerate configurations
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    
    # Compute score
    s = _score(points)
    
    # Penalize if points are not in convex position
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < n_points:
            # Some points are inside the hull, penalize
            penalty = (n_points - len(hull.vertices)) * 0.1
            return -s + penalty
    except Exception:
        return 1e10
    
    # Return negative score (we're minimizing)
    return -s


def _local_refine(points, max_iters=500):
    """
    Local refinement using L-BFGS-B to maximize the minimum triangle area
    while keeping the hull area approximately constant.
    """
    def neg_score(params):
        pts = params.reshape(-1, 2)
        # Normalize to unit hull area
        hull_area = _convex_hull_area(pts)
        if hull_area <= 1e-12:
            return 1e10
        pts_norm = pts / np.sqrt(hull_area)
        s = _score(pts_norm)
        return -s
    
    x0 = points.flatten()
    try:
        result = L-BFGS-B(neg_score, x0, maxiter=max_iters)
        best_pts = result.x.reshape(-1, 2)
        # Normalize
        hull_area = _convex_hull_area(best_pts)
        if hull_area > 1e-12:
            best_pts = best_pts / np.sqrt(hull_area)
        return best_pts
    except Exception:
        return points


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with regular polygon
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Differential evolution for Fourier parameters
    n_fourier_terms = 4
    n_params = 1 + 2 * n_fourier_terms + 1  # a0 + 2*n_coeffs + phase
    
    # Bounds for Fourier coefficients
    bounds = [(0.5, 2.0)]  # a0: mean radius
    for k in range(n_fourier_terms):
        bounds.append((-0.3, 0.3))  # a_k
        bounds.append((-0.3, 0.3))  # b_k
    bounds.append((-np.pi, np.pi))  # phase
    
    # Run differential evolution
    try:
        de_result = differential_evolution(
            _objective_fourier,
            bounds,
            args=(N, n_fourier_terms),
            maxiter=50,
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=False
        )
        
        if de_result.success or de_result.fun < 1e9:
            # Extract best parameters
            best_params = de_result.x
            a0 = best_params[0]
            coeffs = np.array([a0] + list(best_params[1:1 + 2 * n_fourier_terms]))
            phase = best_params[1 + 2 * n_fourier_terms] if len(best_params) > 1 + 2 * n_fourier_terms else 0
            
            theta = np.linspace(0, 2 * np.pi, N, endpoint=False) + phase
            de_points = _fourier_convex_boundary(theta, coeffs)
            
            # Check validity
            de_score = _score(de_points)
            if de_score > best_score:
                best_points = de_points
                best_score = de_score
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        # Not enough time for local refinement
        return best_points
    
    # Local refinement
    try:
        refined_points = _local_refine(best_points, max_iters=200)
        refined_score = _score(refined_points)
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score
    except Exception:
        pass
    
    # If we have more time, do another round of DE with tighter bounds
    if time.time() < deadline - 20:
        try:
            # Tighter bounds around the current best
            center_params = np.array([
                1.0,  # a0
                0.0, 0.0,  # a1, b1
                0.0, 0.0,  # a2, b2
                0.0, 0.0,  # a3, b3
                0.0, 0.0,  # a4, b4
                0.0       # phase
            ])
            
            tight_bounds = [(-0.5, 0.5)] * n_params
            tight_bounds[0] = (0.5, 2.0)  # a0
            tight_bounds[-1] = (-np.pi, np.pi)  # phase
            
            de_result2 = differential_evolution(
                _objective_fourier,
                tight_bounds,
                args=(N, n_fourier_terms),
                maxiter=30,
                popsize=10,
                tol=1e-5,
                seed=43,
                polish=False,
                x0=center_params
            )
            
            if de_result2.fun < 1e9:
                best_params = de_result2.x
                a0 = best_params[0]
                coeffs = np.array([a0] + list(best_params[1:1 + 2 * n_fourier_terms]))
                phase = best_params[1 + 2 * n_fourier_terms] if len(best_params) > 1 + 2 * n_fourier_terms else 0
                
                theta = np.linspace(0, 2 * np.pi, N, endpoint=False) + phase
                de_points2 = _fourier_convex_boundary(theta, coeffs)
                
                de_score2 = _score(de_points2)
                if de_score2 > best_score:
                    best_points = de_points2
                    best_score = de_score2
        except Exception:
            pass
    
    # Final local refinement if time permits
    if time.time() < deadline - 5:
        try:
            refined_points = _local_refine(best_points, max_iters=300)
            refined_score = _score(refined_points)
            if refined_score > best_score:
                best_points = refined_points
                best_score = refined_score
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
