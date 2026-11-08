"""Heilbronn convex regions: place 13 points to maximize minimum triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 13
TIME_LIMIT = 120.0  # seconds


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use polygon area via shoelace if hull fails
        # This shouldn't happen with 13 points
        return 1.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    """Generate vertices of a regular n-gon."""
    angles = 2 * np.pi * np.arange(n) / n + angle_offset
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


def _soft_min(values, k):
    """Soft minimum: -1/k * log(sum(exp(-k * values)))."""
    # For stability, subtract max
    shifted = -k * values
    max_shift = np.max(shifted)
    return -(1.0 / k) * (max_shift + np.log(np.sum(np.exp(shifted - max_shift))))


def construct():
    start_time = time.time()
    
    # Start with regular 13-gon as initial guess
    best_points = _regular_polygon(N, radius=1.0)
    best_score = _score(best_points)
    
    # Parameterize deviations from regular polygon
    # Each point has a radial perturbation and angular perturbation
    # Total parameters: 2*N = 26
    
    def _decode(params):
        """Decode 26 parameters into 13 points."""
        radii = 1.0 + params[:N] * 0.3  # radial perturbation, bounded to [-0.3, 0.3]
        angles = 2 * np.pi * np.arange(N) / N + params[N:] * 0.2  # angular perturbation
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))
    
    def _objective(params):
        """Objective to minimize: negative soft-min of triangle areas / hull area."""
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5:
            return best_score  # Return current best as a constant
        points = _decode(params)
        # Compute all triangle areas
        n = len(points)
        areas = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    v1 = points[j] - points[i]
                    v2 = points[k] - points[i]
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    areas.append(area)
        areas = np.array(areas)
        
        hull_area = _convex_hull_area(points)
        if hull_area <= 1e-12:
            return 1e10
        
        # Use soft minimum with annealing
        elapsed_frac = (time.time() - start_time) / TIME_LIMIT
        # k increases from 5 to 50 as we progress
        k = 5 + 45 * elapsed_frac
        
        soft_min_area = _soft_min(areas, k)
        return -(soft_min_area / hull_area)
    
    # Differential evolution bounds
    bounds = [(-0.3, 0.3)] * N + [(-0.2, 0.2)] * N
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-6,
            popsize=15,
            polish=True,
            init='latinhypercube',
            seed=42
        )
        if result.x is not None:
            candidate_points = _decode(result.x)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
    except Exception:
        pass
    
    # Refine with Nelder-Mead from best point
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 10:
        try:
            def _nm_objective(params):
                elapsed_nm = time.time() - start_time
                if elapsed_nm > TIME_LIMIT - 5:
                    return best_score
                points = _decode(params)
                min_tri = _min_triangle_area(points)
                hull_area = _convex_hull_area(points)
                if hull_area <= 1e-12:
                    return 1e10
                return -(min_tri / hull_area)
            
            # Get current best parameters (we need to invert the decode)
            # Since we don't have a direct inverse, use the result from DE
            if hasattr(result, 'x') and result.x is not None:
                x0 = list(result.x)
            else:
                x0 = [0.0] * (2 * N)
            
            result_nm = minimize(
                _nm_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result_nm.x is not None:
                candidate_points = _decode(result_nm.x)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points
        except Exception:
            pass
    
    # Final check: ensure we return a valid result
    if _score(best_points) <= 0:
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
