"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import itertools
import time

N = 14


def _triangle_areas(pts):
    """Compute all triangle areas from N points (vectorized)."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    return np.array(areas)


def _hull_area(pts):
    """Compute convex hull area of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / hull area."""
    areas = _triangle_areas(pts)
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    return float(np.min(areas) / hull_a)


def _soft_min(areas, beta):
    """Soft minimum using log-sum-exp trick."""
    # -log-sum-exp of -areas / beta gives soft min
    min_val = np.min(areas)
    exp_sum = np.sum(np.exp(-(areas - min_val) / beta))
    return min_val - beta * np.log(exp_sum)


def _soft_score(pts, beta=0.05):
    """Smooth surrogate score using soft-min."""
    areas = _triangle_areas(pts)
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    soft_min_area = _soft_min(areas, beta)
    return float(soft_min_area / hull_a)


def _regular_polygon(n):
    """Return vertices of regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.vstack([np.cos(angles), np.sin(angles)]).T


def _perturb_polygon(pts, seed=0):
    """Perturb regular polygon with random angular/radial noise."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Add small random perturbations to angles and radii
    angle_pert = rng.uniform(-0.1, 0.1, N)
    radius_pert = rng.uniform(0.9, 1.1, N)
    new_angles = angles + angle_pert
    new_radii = radius_pert
    new_pts = np.vstack([new_radii * np.cos(new_angles),
                         new_radii * np.sin(new_angles)]).T
    return new_pts


def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer before 120s kill

    # Start with regular 14-gon
    best_pts = _regular_polygon(N)
    best_score = _score(best_pts)

    # Phase 1: Differential Evolution with soft-min surrogate
    def _de_objective(x):
        """DE objective: negative soft score (we maximize, DE minimizes)."""
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            return 0.0  # Timeout guard
        pts = x.reshape(N, 2)
        # Center and normalize
        pts = pts - np.mean(pts, axis=0)
        # Normalize to unit area hull
        hull_a = _hull_area(pts)
        if hull_a <= 0:
            return 0.0
        pts = pts / np.sqrt(hull_a)
        return -_soft_score(pts, beta=0.03)

    # Bounds: points in [-2, 2] x [-2, 2]
    bounds = [(-2.0, 2.0)] * (2 * N)
    x0 = best_pts.flatten()

    try:
        de_result = differential_evolution(
            _de_objective,
            bounds=bounds,
            seed=42,
            maxiter=80,
            tol=1e-5,
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,
            init='sohal'
        )
        if de_result.success or _de_objective(de_result.x) < _de_objective(x0):
            cand = de_result.x.reshape(N, 2)
            cand = cand - np.mean(cand, axis=0)
            hull_a = _hull_area(cand)
            if hull_a > 0:
                cand = cand / np.sqrt(hull_a)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
    except Exception:
        pass

    # Phase 2: Multi-restart local optimization (Powell + Nelder-Mead)
    def _local_objective(x):
        """Local objective: negative exact score."""
        elapsed = time.time() - start_time
        if elapsed > time_limit - 3:
            return 0.0
        pts = x.reshape(N, 2)
        pts = pts - np.mean(pts, axis=0)
        hull_a = _hull_area(pts)
        if hull_a <= 0:
            return 0.0
        pts = pts / np.sqrt(hull_a)
        return -_score(pts)

    # Try multiple restarts from best and perturbed points
    rng = np.random.RandomState(123)
    for restart in range(20):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            break

        if restart == 0:
            x0 = best_pts.flatten()
        else:
            # Perturb best point
            x0 = best_pts + rng.normal(0, 0.05, (N, 2))
            x0 = x0.flatten()

        try:
            # Powell method
            res_powell = minimize(
                _local_objective,
                x0=x0,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            if res_powell.fun < -best_score:
                cand = res_powell.x.reshape(N, 2)
                cand = cand - np.mean(cand, axis=0)
                hull_a = _hull_area(cand)
                if hull_a > 0:
                    cand = cand / np.sqrt(hull_a)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand

            # Nelder-Mead polish
            elapsed = time.time() - start_time
            if elapsed > time_limit - 5:
                break
            res_nm = minimize(
                _local_objective,
                x0=best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_nm.fun < -best_score:
                cand = res_nm.x.reshape(N, 2)
                cand = cand - np.mean(cand, axis=0)
                hull_a = _hull_area(cand)
                if hull_a > 0:
                    cand = cand / np.sqrt(hull_a)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
        except Exception:
            continue

    # Final: return best points (already normalized to unit hull area)
    return best_pts
