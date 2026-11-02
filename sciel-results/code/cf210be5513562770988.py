import time
import numpy as np
from scipy.optimize import basinhopping, minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for triangles to speed up min area calculation
# There are C(14, 3) = 364 triangles
# We can store the indices (i, j, k)
_TRI_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRI_IDX.append((i, j, k))
_TRI_IDX = np.array(_TRI_IDX)

def _min_triangle_area(points):
    """
    Compute the minimum area over all triples of points.
    points: (N, 2) array
    """
    # Vectorized calculation of cross products for all triangles
    # p_i, p_j, p_k
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    
    # Get points for each triangle
    pi = points[_TRI_IDX[:, 0]]
    pj = points[_TRI_IDX[:, 1]]
    pk = points[_TRI_IDX[:, 2]]
    
    # v1 = pj - pi, v2 = pk - pi
    v1 = pj - pi
    v2 = pk - pi
    
    # Cross product (z-component)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    # Area is 0.5 * abs(cross)
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        # Check if points are collinear or fewer than 3
        if len(points) < 3:
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Compute the ratio: min triangle area / convex hull area.
    """
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _objective(x):
    """
    Objective function for scipy optimizers.
    x: 1D array of length 2*N (or N for radial)
    Returns negative score (to be minimized).
    """
    # Determine dimensionality
    if len(x) == N:
        # Radial parameterization
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points = np.column_stack([x * np.cos(angles), x * np.sin(angles)])
    else:
        points = x.reshape(-1, 2)
    
    # Ensure points are finite
    if not np.all(np.isfinite(points)):
        return 1e10
        
    s = _score(points)
    return -s

# Fixed angles for radial parameterization
_ANGLES_RAD = np.linspace(0, 2 * np.pi, N, endpoint=False)

def _points_from_radii(radii):
    """Convert radial coordinates to Cartesian points."""
    x = radii * np.cos(_ANGLES_RAD)
    y = radii * np.sin(_ANGLES_RAD)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    # 150s limit, use 145s to be safe
    deadline = start_time + 145.0
    
    # Initial guess: regular 14-gon on unit circle
    r0 = np.ones(N)
    x0_full = _points_from_radii(r0).flatten()
    
    best_x = x0_full.copy()
    best_score = _score(best_x.reshape(-1, 2))
    
    # Helper to check time
    def time_check():
        return time.time() > deadline - 10.0  # Leave 10s for final polish/return

    # --- Phase 1: Radial Optimization (14D) ---
    # This finds a good convex shape
    try:
        bh_radial = basinhopping(
            _objective,
            r0,
            minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}},
            niter=50,
            stepsize=0.5,
            callback=lambda *args, **kwargs: time_check(),
            seed=42
        )
        if bh_radial.fun < -best_score:
            r_best = bh_radial.x
            points_best = _points_from_radii(r_best)
            best_score = _score(points_best)
            best_x = points_best.flatten()
    except Exception:
        pass

    # --- Phase 2: Full Coordinate Optimization (28D) ---
    # Start from the best radial result and allow points to move off the rays
    # This is crucial for breaking the symmetry of the regular polygon/radial constraint
    if time.time() < deadline - 20.0:
        try:
            bh_full = basinhopping(
                _objective,
                best_x,
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}},
                niter=50,
                stepsize=0.1, # Smaller steps for 28D
                callback=lambda *args, **kwargs: time_check(),
                seed=123
            )
            if bh_full.fun < -best_score:
                best_x = bh_full.x
                best_score = _score(best_x.reshape(-1, 2))
        except Exception:
            pass

    # --- Phase 3: Final Polish ---
    # Use a high-precision local optimization
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                _objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < -best_score:
                best_x = res.x
                best_score = _score(best_x.reshape(-1, 2))
        except Exception:
            pass

    return best_x.reshape(-1, 2)
