"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import differential_evolution

DEADLINE = 140.0  # safety margin before 150s kill
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    # Use Delaunay triangulation to only check adjacent triangles
    try:
        tri = Delaunay(points)
        for simplex in tri.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
            if area < min_area:
                min_area = area
    except Exception:
        pass
    # Also check all triples if Delaunay failed or for safety
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(points):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: shoelace on all points (assume convex)
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i, 0] * points[j, 1] - points[j, 0] * points[i, 1]
        return abs(area) / 2.0


def _score(points):
    """Score = min triangle area / hull area."""
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0
    mta = _min_triangle_area(points)
    return mta / ha


def _two_ring_params_to_points(params):
    """Convert two-ring parameters to 14 points.
    
    params: [r1, r2, theta1, theta2, r_inner, n_inner, offset_inner, ...]
    We use a structured approach:
    - 7 points on outer ring (radius r1, angles theta1 + k*2pi/7)
    - 7 points on inner ring (radius r2, angles theta2 + k*2pi/7 + offset)
    
    params: [r1, r2, theta1, theta2, offset, r1_var, r2_var, phase_var]
    Actually let's use simpler: 
    params: [r1, r2, theta1, theta2, offset, jitter_scale]
    """
    r1 = params[0]
    r2 = params[1]
    theta1 = params[2]
    theta2 = params[3]
    offset = params[4]
    
    n_outer = 7
    n_inner = 7
    
    points = np.zeros((N, 2))
    # Outer ring
    for k in range(n_outer):
        angle = theta1 + 2 * np.pi * k / n_outer
        points[2 * k, 0] = r1 * np.cos(angle)
        points[2 * k, 1] = r1 * np.sin(angle)
    # Inner ring
    for k in range(n_inner):
        angle = theta2 + offset + 2 * np.pi * k / n_inner
        points[2 * k + 1, 0] = r2 * np.cos(angle)
        points[2 * k + 1, 1] = r2 * np.sin(angle)
    
    return points


def _objective_two_ring(params):
    points = _two_ring_params_to_points(params)
    return -_score(points)


def _three_ring_params_to_points(params):
    """Three-ring structure: 5 + 5 + 4 points on three concentric rings."""
    r1 = params[0]
    r2 = params[1]
    r3 = params[2]
    theta1 = params[3]
    theta2 = params[4]
    theta3 = params[5]
    offset2 = params[6]
    offset3 = params[7]
    
    points = np.zeros((N, 2))
    # Ring 1: 5 points
    for k in range(5):
        angle = theta1 + 2 * np.pi * k / 5
        points[k, 0] = r1 * np.cos(angle)
        points[k, 1] = r1 * np.sin(angle)
    # Ring 2: 5 points
    for k in range(5):
        angle = theta2 + offset2 + 2 * np.pi * k / 5
        points[5 + k, 0] = r2 * np.cos(angle)
        points[5 + k, 1] = r2 * np.sin(angle)
    # Ring 3: 4 points
    for k in range(4):
        angle = theta3 + offset3 + 2 * np.pi * k / 4
        points[10 + k, 0] = r3 * np.cos(angle)
        points[10 + k, 1] = r3 * np.sin(angle)
    
    return points


def _objective_three_ring(params):
    points = _three_ring_params_to_points(params)
    return -_score(points)


def _single_ring_params_to_points(params):
    """All 14 points on a single circle with variable angular spacing.
    params: [radius, theta0, ... 13 angle offsets relative to uniform]
    Actually, use: [radius, theta0, d1, d2, ..., d13] where dk are deviations
    from uniform spacing, with sum(dk)=0 constraint handled by parameterization.
    Simpler: [radius, theta0, a1, a2, ..., a13] where angles are cumulative.
    """
    radius = params[0]
    theta0 = params[1]
    # 13 free angle parameters, 14th determined by closure
    # Use: angles = theta0 + cumsum([d1, d2, ..., d13, 2pi - sum(d1..d13)])
    d = params[2:15]
    angles = np.zeros(14)
    angles[0] = theta0
    for k in range(1, 14):
        angles[k] = angles[k-1] + d[k-1]
    # Normalize to ensure they span 2pi
    # Actually, let's just use uniform + perturbations
    # Better: use Fourier-like parameterization
    # angles[k] = theta0 + 2pi*k/14 + sum_{m=1}^{6} (a_m cos(2pi*m*k/14) + b_m sin(2pi*m*k/14))
    # But that's 12 params + radius + theta0 = 14 params
    # Let's use: params = [radius, theta0, a1, b1, a2, b2, ..., a6, b6]
    # That's 2 + 12 = 14 params
    # But we only have 14 params total, so this works
    # Reinterpret params
    # Actually let's just do uniform circle for simplicity in this function
    # and rely on the multi-ring structures
    points = np.zeros((N, 2))
    for k in range(N):
        angle = theta0 + 2 * np.pi * k / N
        points[k, 0] = radius * np.cos(angle)
        points[k, 1] = radius * np.sin(angle)
    return points


def _objective_single_ring(params):
    points = _single_ring_params_to_points(params)
    return -_score(points)


def construct():
    global _start_time
    _start_time = time.time()
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: uniform circle
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(theta), np.sin(theta)])
    best_score = _score(init_points)
    best_points = init_points.copy()
    
    # Try two-ring structure with DE
    bounds_two = [(0.3, 1.0), (0.0, 0.8), (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
    try:
        if _elapsed() < DEADLINE - 5:
            res = differential_evolution(
                _objective_two_ring, bounds_two,
                seed=42, maxiter=100, popsize=15,
                tol=1e-6, polish=True,
                callback=lambda x, convergence: (np.inf if _elapsed() > DEADLINE - 5 else None)
            )
            if res.success or res.fun < float('inf'):
                pts = _two_ring_params_to_points(res.x)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
    except Exception:
        pass
    
    # Try three-ring structure with DE
    bounds_three = [(0.2, 1.0), (0.0, 0.7), (0.0, 0.4),
                    (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi),
                    (0, 2*np.pi), (0, 2*np.pi)]
    try:
        if _elapsed() < DEADLINE - 5:
            res = differential_evolution(
                _objective_three_ring, bounds_three,
                seed=123, maxiter=80, popsize=12,
                tol=1e-6, polish=True,
                callback=lambda x, convergence: (np.inf if _elapsed() > DEADLINE - 5 else None)
            )
            if res.success or res.fun < float('inf'):
                pts = _three_ring_params_to_points(res.x)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
    except Exception:
        pass
    
    # Try single ring (uniform circle) as baseline
    try:
        if _elapsed() < DEADLINE - 5:
            res = differential_evolution(
                _objective_single_ring, [(0.5, 1.0), (0, 2*np.pi)] + [(0, 2*np.pi)]*13,
                seed=7, maxiter=50, popsize=10,
                tol=1e-6, polish=True,
                callback=lambda x, convergence: (np.inf if _elapsed() > DEADLINE - 5 else None)
            )
            if res.success or res.fun < float('inf'):
                pts = _single_ring_params_to_points(res.x)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
    except Exception:
        pass
    
    # Final polish: local refinement using Nelder-Mead on the best point set
    try:
        if _elapsed() < DEADLINE - 3 and best_points is not None:
            from scipy.optimize import minimize
            
            def _neg_score_flat(flat):
                pts = flat.reshape(N, 2)
                return -_score(pts)
            
            res = minimize(
                _neg_score_flat, best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                         'callback': lambda x: (None if _elapsed() < DEADLINE - 2 else np.inf)}
            )
            if res.fun < float('inf'):
                pts = res.x.reshape(N, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
    except Exception:
        pass
    
    if best_points is None:
        # Fallback: uniform circle
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(theta), np.sin(theta)])
    
    return best_points
# EVOLVE-BLOCK-END
