import numpy as np

N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return 1.0

def _score(points):
    """Compute score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _params_to_points(params):
    """Convert (angles, radii) to Cartesian points.
    params: array of shape (2*N,) where first N are angles, next N are radii.
    Angles are sorted to maintain ordering, radii are positive.
    """
    angles = np.sort(params[:N])
    radii = np.abs(params[N:]) + 0.1  # ensure positive radii
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

def _objective(params):
    """Objective function for optimization (negative score)."""
    points = _params_to_points(params)
    return -_score(points)

def _regular_polygon_params(n, radius=1.0):
    """Generate regular n-gon in (angles, radii) parameter space."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.full(n, radius)
    return np.concatenate([angles, radii])

def construct():
    """
    Constructs 13 points to maximize the minimum triangle area relative to hull area.
    Uses differential evolution over polar parameters, then L-BFGS-B polish.
    """
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline for safety
    
    # Initial guess: regular 13-gon in polar space
    x0 = _regular_polygon_params(N, radius=1.0)
    best_points = _params_to_points(x0)
    best_score = _score(best_points)
    
    # Strategy 1: Differential Evolution over polar parameters
    if time.time() < deadline - 60:
        # Bounds: angles in [0, 2*pi), radii in [0.3, 2.0]
        bounds = [(0, 2*np.pi)] * N + [(0.3, 2.0)] * N
        
        try:
            result = differential_evolution(
                _objective, bounds,
                seed=42, maxiter=100, tol=1e-8,
                popsize=15, mutation=(0.3, 1.0), recombination=0.7,
                polish=True, init='latinhypercube',
                updating='immediate'
            )
            
            if time.time() < deadline - 30:
                candidate_points = _params_to_points(result.x)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_points = candidate_points
                    best_score = candidate_score
        except Exception:
            pass
    
    # Strategy 2: L-BFGS-B polish from best solution
    if time.time() < deadline - 20:
        x0 = _params_to_points(best_points)  # Use Cartesian for polish
        # Convert back to polar for L-BFGS-B
        cx = np.mean(x0[:, 0])
        cy = np.mean(x0[:, 1])
        angles = np.arctan2(x0[:, 1] - cy, x0[:, 0] - cx)
        radii = np.sqrt((x0[:, 0] - cx)**2 + (x0[:, 1] - cy)**2)
        # Sort by angle
        sort_idx = np.argsort(angles)
        angles = angles[sort_idx]
        radii = radii[sort_idx]
        # Make angles continuous (unwind)
        for i in range(1, N):
            if angles[i] - angles[i-1] < 0:
                angles[i:] += 2*np.pi
        
        x0_polar = np.concatenate([angles, radii])
        
        # Objective in Cartesian space for L-BFGS-B
        def _cart_obj(x):
            pts = x.reshape(N, 2)
            return -_score(pts)
        
        try:
            result = minimize(_cart_obj, best_points.flatten(), method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if time.time() < deadline - 10:
                candidate_points = result.x.reshape(N, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_points = candidate_points
                    best_score = candidate_score
        except Exception:
            pass
    
    # Strategy 3: Second DE run with different seed if time allows
    if time.time() < deadline - 40:
        bounds = [(0, 2*np.pi)] * N + [(0.3, 2.0)] * N
        try:
            result = differential_evolution(
                _objective, bounds,
                seed=123, maxiter=80, tol=1e-8,
                popsize=12, mutation=(0.5, 1.0), recombination=0.8,
                polish=False, init='sobol',
                updating='immediate'
            )
            
            if time.time() < deadline - 10:
                candidate_points = _params_to_points(result.x)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_points = candidate_points
                    best_score = candidate_score
        except Exception:
            pass
    
    # Final polish: Nelder-Mead on Cartesian from best
    if time.time() < deadline - 5:
        try:
            def _nm_obj(x):
                pts = x.reshape(N, 2)
                return -_score(pts)
            
            result = minimize(_nm_obj, best_points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12,
                                      'adaptive': True, 'maxfev': 5000})
            
            candidate_points = result.x.reshape(N, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END

