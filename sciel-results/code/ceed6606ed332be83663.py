"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                # Area = 0.5 * |cross(b-a, c-a)|
                ab = b - a
                ac = c - a
                area = 0.5 * abs(ab[0] * ac[1] - ab[1] * ac[0])
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if points are collinear or degenerate
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_a = _hull_area(points)
    if hull_a <= 0:
        return 0.0
    min_ta = _min_triangle_area(points)
    return min_ta / hull_a


def _is_convex_position(points):
    """Check if points are in convex position (all on convex hull)."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return len(hull.vertices) == len(points)
    except Exception:
        return False


def _negative_min_triangle_area(points_flat):
    """Objective function for optimizer: minimize negative min triangle area.
    
    We also add a penalty if points are not in convex position.
    """
    n = len(points_flat)
    points = points_flat.reshape(-1, 2)
    
    # Check convex position
    if not _is_convex_position(points):
        # Large penalty for non-convex position
        return 1000.0
    
    min_ta = _min_triangle_area(points)
    return -min_ta


def _regular_polygon(n, scale=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return scale * np.column_stack([np.cos(angles), np.sin(angles)])


def construct():
    """
    Place 13 points to maximize the minimum triangle area divided by convex hull area.
    
    Strategy:
    1. Start with a regular 13-gon.
    2. Refine using local optimization (L-BFGS-B) to maximize min triangle area
       while maintaining convex position.
    3. Use multiple random perturbations + local refinement to explore the landscape.
    4. Return the best configuration found within the time budget.
    """
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Precompute all triple indices for efficiency
    triples = list(combinations(range(N), 3))
    
    def _min_area_fast(points):
        """Fast min triangle area using precomputed triples."""
        min_a = float('inf')
        for i, j, k in triples:
            a = points[i]
            b = points[j]
            c = points[k]
            ab = b - a
            ac = c - a
            area = 0.5 * abs(ab[0] * ac[1] - ab[1] * ac[0])
            if area < min_a:
                min_a = area
        return min_a
    
    def _neg_min_area_fast(points_flat):
        """Fast objective for optimizer."""
        points = points_flat.reshape(-1, 2)
        # Quick convexity check via cross products of consecutive edges
        # Sort by angle around centroid for convex position check
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        order = np.argsort(angles)
        ordered = points[order]
        
        # Check all consecutive cross products are positive (CCW)
        n_pts = len(ordered)
        for idx in range(n_pts):
            p0 = ordered[idx]
            p1 = ordered[(idx + 1) % n_pts]
            p2 = ordered[(idx + 2) % n_pts]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if cross <= 1e-10:
                return 1000.0  # Penalty
        
        return -_min_area_fast(points)
    
    # Try optimization from regular polygon
    def _optimize_from(start_points):
        """Run local optimization from a starting configuration."""
        x0 = start_points.flatten()
        
        # Use L-BFGS-B with bounds to keep points in a reasonable region
        # No strict bounds needed if we penalize non-convex positions
        result = minimize(
            _neg_min_area_fast,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < 100:
            opt_points = result.x.reshape(-1, 2)
            if _is_convex_position(opt_points):
                return opt_points
        return None
    
    # Phase 1: Optimize from regular polygon
    if _time_remaining() > 30:
        opt = _optimize_from(best_points)
        if opt is not None:
            s = _score(opt)
            if s > best_score:
                best_score = s
                best_points = opt
    
    # Phase 2: Try random perturbations + local optimization
    np.random.seed(42)
    n_trials = 20
    for trial in range(n_trials):
        if _time_remaining() < 15:
            break
        
        # Random perturbation of the regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + 0.1 * np.random.randn(N)
        radii = np.maximum(radii, 0.5)  # Keep positive
        perturbed = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Also try random angular perturbations
        angle_pert = angles + 0.05 * np.random.randn(N)
        perturbed2 = np.column_stack([np.cos(angle_pert), np.sin(angle_pert)])
        
        for start in [perturbed, perturbed2]:
            if _time_remaining() < 10:
                break
            opt = _optimize_from(start)
            if opt is not None:
                s = _score(opt)
                if s > best_score:
                    best_score = s
                    best_points = opt
    
    # Phase 3: Try a few more diverse starting points
    if _time_remaining() > 20:
        # Try points on different shapes
        for shape_type in ['ellipse', 'random_convex']:
            if _time_remaining() < 10:
                break
            
            if shape_type == 'ellipse':
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                points_ellipse = np.column_stack([1.5 * np.cos(angles), 0.8 * np.sin(angles)])
                opt = _optimize_from(points_ellipse)
            elif shape_type == 'random_convex':
                # Random convex polygon vertices
                angles = np.sort(np.random.uniform(0, 2 * np.pi, N))
                radii = np.random.uniform(0.8, 1.2, N)
                points_random = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
                opt = _optimize_from(points_random)
            else:
                opt = None
            
            if opt is not None:
                s = _score(opt)
                if s > best_score:
                    best_score = s
                    best_points = opt
    
    # Final: ensure we return a valid convex position
    if not _is_convex_position(best_points):
        # Fallback to regular polygon
        best_points = _regular_polygon(N)
    
    return best_points
# EVOLVE-BLOCK-END
