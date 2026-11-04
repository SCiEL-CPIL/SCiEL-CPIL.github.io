"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools

def _min_triangle_area_vectorized(points):
    """Compute the minimum area over all triples of points using vectorization."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate all combinations of 3 indices
    # C(14,3) = 364, which is small enough to materialize
    indices = list(itertools.combinations(range(n), 3))
    if not indices:
        return 0.0
    
    # Extract points for each triple
    # Shape: (num_triples, 3, 2)
    triples = np.array([points[list(idx)] for idx in indices])
    
    # Calculate area for each triple
    # v1 = p1 - p0, v2 = p2 - p0
    # Area = 0.5 * |v1_x * v2_y - v1_y * v2_x|
    p0 = triples[:, 0, :]
    p1 = triples[:, 1, :]
    p2 = triples[:, 2, :]
    
    v1 = p1 - p0
    v2 = p2 - p0
    
    cross_prod = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross_prod)
    
    return np.min(areas)

def _convex_hull_area_fast(points):
    """Compute the area of the convex hull using scipy."""
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        return 0.0

def _score_fast(points):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area_vectorized(points)
    hull_a = _convex_hull_area_fast(points)
    if hull_a <= 1e-10:
        return 0.0
    return min_ta / hull_a

def _build_points_from_params(params):
    """
    params: 24-dim array
    [0:7]   = angular offsets for 7 heptagon vertices (radians)
    [7:21]  = radii for 14 points (7 pairs, 2 per pair)
    [21]    = rotation angle
    [22]    = scale
    [23]    = shear
    """
    angle_offsets = params[:7]
    radii = params[7:21]
    rot = params[21]
    scale = params[22]
    shear = params[23]
    
    # Base angles for regular heptagon
    base_angles = np.arange(7) * (2 * np.pi / 7)
    
    # Actual angles
    angles = base_angles + angle_offsets
    
    points = np.zeros((14, 2))
    
    for i in range(7):
        r1 = radii[2*i]
        r2 = radii[2*i+1]
        # Ensure positive radii
        r1 = max(r1, 0.1)
        r2 = max(r2, 0.1)
        
        a = angles[i]
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        
        points[2*i] = [r1 * cos_a, r1 * sin_a]
        points[2*i+1] = [r2 * cos_a, r2 * sin_a]
    
    # Apply affine transformation
    c = np.cos(rot)
    s = np.sin(rot)
    # Matrix: [[scale*c, -scale*s + shear], [scale*s, scale*c]]
    # Note: Standard shear is usually added to one axis. 
    # Let's use a simple affine transform: x' = scale*x*cos(rot) - scale*y*sin(rot) + shear*x? 
    # Or just rotation + scale + shear in x.
    # A = [[scale*c, -scale*s], [scale*s, scale*c]] + [[shear, 0], [0, 0]]?
    # Let's stick to the previous definition for consistency:
    # A = [[scale*c, -scale*s + shear], [scale*s, scale*c]]
    A = np.array([[scale * c, -scale * s + shear],
                  [scale * s,  scale * c]])
    
    points = points @ A.T
    
    return points

def _negative_score(params):
    """Objective function for minimization."""
    points = _build_points_from_params(params)
    score = _score_fast(points)
    return -score

def construct():
    """
    Construct 14 points by optimizing 24 parameters:
    7 angular offsets, 14 radii, 3 affine parameters.
    """
    # Bounds for parameters
    # Angles: small offsets from regular heptagon
    angle_bounds = [(-0.2, 0.2) for _ in range(7)]
    # Radii: between 0.5 and 1.5
    radius_bounds = [(0.5, 1.5) for _ in range(14)]
    # Affine
    rot_bounds = [(0, 2*np.pi)]
    scale_bounds = [(0.5, 2.0)]
    shear_bounds = [(-0.5, 0.5)]
    
    bounds = angle_bounds + radius_bounds + rot_bounds + scale_bounds + shear_bounds
    
    # Initial guess: Regular heptagon structure
    x0 = np.zeros(24)
    x0[7:21] = 1.0 # Radii
    x0[22] = 1.0 # Scale
    
    # Global optimization using Differential Evolution
    # This is robust for non-convex problems
    try:
        result = differential_evolution(
            _negative_score, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-6,
            polish=False # We will polish manually
        )
        best_params = result.x
        best_score = -result.fun
    except Exception:
        best_params = x0
        best_score = -_negative_score(x0)

    # Local polish using Nelder-Mead
    try:
        res = minimize(_negative_score, best_params, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        if -res.fun > best_score:
            best_params = res.x
            best_score = -res.fun
    except Exception:
        pass

    points = _build_points_from_params(best_params)
    
    # Fallback check
    if _convex_hull_area_fast(points) <= 1e-10:
        # Fallback to a known valid configuration
        base_angles = np.arange(7) * (2 * np.pi / 7)
        points = np.zeros((14, 2))
        for i in range(7):
            points[2*i] = [1.1 * np.cos(base_angles[i]), 1.1 * np.sin(base_angles[i])]
            points[2*i+1] = [0.9 * np.cos(base_angles[i]), 0.9 * np.sin(base_angles[i])]
    
    return points
# EVOLVE-BLOCK-END
