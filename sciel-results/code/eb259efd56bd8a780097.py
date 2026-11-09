# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools


def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(pts):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    if len(pts) < 3:
        return 0.0
    a_min = _min_triangle_area(pts)
    a_hull = _hull_area(pts)
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull


def _angles_to_points(angles, radius=1.0):
    """Convert angles on a circle to 2D points."""
    angles = np.asarray(angles)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack([x, y])


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 570  # Leave 60s margin from 630s to be safe

    # Start with regular polygon as initial guess
    initial_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    def neg_score(angles):
        """Negative of score for minimization."""
        pts = _angles_to_points(angles, radius=1.0)
        s = _score(pts)
        return -s

    # Phase 1: Differential evolution
    bounds = [(0, 2 * np.pi)] * n
    best_result = None
    best_score_val = -np.inf
    best_angles = initial_angles.copy()

    try:
        de_result = differential_evolution(
            neg_score,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-10,
            polish=False,
            init='sobol',
            update_interval=1,
            callback=lambda x, convergence: None
        )
        if de_result.success or de_result.fun < best_score_val:
            best_angles = de_result.x
            best_score_val = -de_result.fun
            best_result = _angles_to_points(best_angles, radius=1.0)
    except Exception:
        pass

    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit - 30:
        if best_result is not None:
            return best_result
        return _angles_to_points(initial_angles, radius=1.0)

    # Phase 2: Local refinement with Nelder-Mead
    try:
        nm_result = minimize(
            neg_score,
            x0=best_angles,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        if nm_result.fun < best_score_val:
            best_angles = nm_result.x
            best_score_val = -nm_result.fun
            best_result = _angles_to_points(best_angles, radius=1.0)
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > time_limit - 10:
        if best_result is not None:
            return best_result
        return _angles_to_points(initial_angles, radius=1.0)

    # Phase 3: Try a few different radii and slight perturbations
    if best_result is not None:
        final_pts = best_result
        final_score = _score(final_pts)

        # Try perturbing and re-optimizing
        rng = np.random.default_rng(seed=123)
        for trial in range(5):
            elapsed = time.time() - start_time
            if elapsed > time_limit - 20:
                break
            perturbed = best_angles + rng.normal(0, 0.1, n)
            perturbed = perturbed % (2 * np.pi)
            try:
                nm2 = minimize(
                    neg_score,
                    x0=perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                if nm2.fun < best_score_val:
                    best_angles = nm2.x
                    best_score_val = -nm2.fun
                    final_pts = _angles_to_points(best_angles, radius=1.0)
            except Exception:
                pass

        if final_score <= 0:
            return _angles_to_points(initial_angles, radius=1.0)
        return final_pts

    return _angles_to_points(initial_angles, radius=1.0)


# EVOLVE-BLOCK-END
