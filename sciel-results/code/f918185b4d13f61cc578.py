"""Optimized program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull


def _hull_area(pts):
    """Compute convex hull area, with fallback to 1.0."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1.0


def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                a = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if a < min_a:
                    min_a = a
    return min_a


def _score(pts):
    """Score = min triangle area / hull area."""
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / ha


def _objective_neg(pts):
    """Negative score for minimization."""
    return -_score(pts)


def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit

    # Start with regular polygon
    best_pts = _regular_polygon(N)
    best_score = _score(best_pts)

    # Try a few initializations and local optimizations
    # Parameterize points on a circle: angles theta_i
    # This keeps points on convex hull (circle), hull = circle area
    # But we can also allow radial variation for non-circular convex shapes

    # Strategy 1: Optimize angles on a unit circle
    # Points: (cos(theta_i), sin(theta_i))
    # Hull area = area of convex hull of these points
    # For points on a circle in order, hull is the polygon itself

    def angles_to_pts(angles):
        angles = np.sort(angles)  # ensure convex order
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def neg_score_from_angles(angles):
        pts = angles_to_pts(angles)
        return -_score(pts)

    # Initial: equally spaced
    x0 = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Use differential evolution for global search, then polish
    # But DE might be slow; use a few random restarts with local optimization

    best_score = _score(best_pts)

    # Local optimization on angles using L-BFGS-B
    from scipy.optimize import minimize

    def neg_obj_angles(angles):
        return neg_score_from_angles(angles)

    # Try multiple random perturbations of the regular polygon
    np.random.seed(42)
    for trial in range(8):
        if time.time() > deadline:
            break

        if trial == 0:
            x0_trial = x0.copy()
        else:
            # Random perturbation
            x0_trial = x0 + np.random.uniform(-0.1, 0.1, N)
            x0_trial = np.sort(x0_trial)

        try:
            res = minimize(neg_obj_angles, x0_trial, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10})
            if res.fun < -best_score:
                best_score = -res.fun
                best_pts = angles_to_pts(res.x)
        except Exception:
            pass

    # Strategy 2: Optimize full 2D positions with convexity constraint
    # Use a different parameterization: support function or just 2D with penalty
    # For simplicity, try optimizing 2D positions directly with a convexity check

    def neg_obj_2d(x):
        pts = x.reshape(N, 2)
        # Check if points are in convex position (all on hull)
        try:
            hull = ConvexHull(pts)
            if len(hull.vertices) < N - 1:  # Some points are interior
                return -best_score - 100  # Penalize
        except Exception:
            return -best_score - 100
        return -_score(pts)

    # Start from best found
    x0_2d = best_pts.flatten()
    try:
        res2 = minimize(neg_obj_2d, x0_2d, method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-12})
        if res2.fun < -best_score:
            best_score = -res2.fun
            best_pts = res2.x.reshape(N, 2)
    except Exception:
        pass

    # Final polish: try a few more local optimizations from best
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline:
                break
            try:
                res3 = minimize(neg_obj_2d, best_pts.flatten() + np.random.randn(N, 2) * 0.01,
                                method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12})
                if res3.fun < -best_score:
                    best_score = -res3.fun
                    best_pts = res3.x.reshape(N, 2)
            except Exception:
                pass

    return best_pts
# EVOLVE-BLOCK-END
