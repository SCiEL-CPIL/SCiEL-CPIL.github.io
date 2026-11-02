"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay
import time
import itertools
import random

DEADLINE = 120.0  # seconds, with safety margin


def _hull_area(pts):
    """Compute convex hull area of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples using vectorized computation."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _score(pts):
    """Score: min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    mta = _min_triangle_area(pts)
    return mta / ha


def _params_to_points(params):
    """
    Convert parameter vector to 13 points.
    params = [cx, cy, r1, r2, r3, a1, a2, a3, p1, p2, p3, q1, q2, q3, s1, s2, s3]
    where:
      cx, cy: center
      r1, r2, r3: ring radii
      a1, a2, a3: ring angular offsets
      p1, p2, p3: per-ring angular spacing modulation (0 = regular, >0 = perturbed)
      q1, q2, q3: per-ring radius modulation
      s1, s2, s3: per-ring angular perturbation amplitude
    """
    cx, cy = params[0], params[1]
    r1, r2, r3 = params[2], params[3], params[4]
    a1, a2, a3 = params[5], params[6], params[7]
    p1, p2, p3 = params[8], params[9], params[10]
    q1, q2, q3 = params[11], params[12], params[13]
    s1, s2, s3 = params[14], params[15], params[16]

    pts = np.empty((13, 2))
    pts[0] = [cx, cy]

    # Ring 1: 4 points
    for m in range(4):
        base_angle = a1 + m * (2 * np.pi / 4)
        # Angular perturbation
        angle = base_angle + s1 * np.sin(p1 * m)
        # Radius perturbation
        rad = r1 * (1.0 + q1 * np.cos(p2 * m))
        pts[1 + m, 0] = cx + rad * np.cos(angle)
        pts[1 + m, 1] = cy + rad * np.sin(angle)

    # Ring 2: 4 points
    for m in range(4):
        base_angle = a2 + m * (2 * np.pi / 4)
        angle = base_angle + s2 * np.sin(p1 * m + 0.5)
        rad = r2 * (1.0 + q2 * np.cos(p3 * m + 1.0))
        pts[5 + m, 0] = cx + rad * np.cos(angle)
        pts[5 + m, 1] = cy + rad * np.sin(angle)

    # Ring 3: 4 points
    for m in range(4):
        base_angle = a3 + m * (2 * np.pi / 4)
        angle = base_angle + s3 * np.sin(p2 * m + 1.0)
        rad = r3 * (1.0 + q3 * np.cos(p1 * m + 2.0))
        pts[9 + m, 0] = cx + rad * np.cos(angle)
        pts[9 + m, 1] = cy + rad * np.sin(angle)

    return pts


def _score_params(params):
    """Score from parameter vector."""
    pts = _params_to_points(params)
    return _score(pts)


def _score_params_neg(params):
    """Negative score for minimization."""
    return -_score_params(params)


def _generate_ring_starts(n_starts, rng):
    """Generate diverse ring parameter initializations."""
    starts = []
    for i in range(n_starts):
        # Center near 0.5 with small perturbation
        cx = 0.5 + rng.uniform(-0.05, 0.05)
        cy = 0.5 + rng.uniform(-0.05, 0.05)

        # Ring radii: roughly 0.2, 0.35, 0.45 with variation
        r1 = rng.uniform(0.15, 0.28)
        r2 = rng.uniform(0.30, 0.40)
        r3 = rng.uniform(0.40, 0.50)

        # Angular offsets
        a1 = rng.uniform(0, 2 * np.pi)
        a2 = rng.uniform(0, 2 * np.pi)
        a3 = rng.uniform(0, 2 * np.pi)

        # Perturbation parameters (small)
        p1 = rng.uniform(1.0, 3.0)
        p2 = rng.uniform(1.0, 3.0)
        p3 = rng.uniform(1.0, 3.0)
        q1 = rng.uniform(-0.1, 0.1)
        q2 = rng.uniform(-0.1, 0.1)
        q3 = rng.uniform(-0.1, 0.1)
        s1 = rng.uniform(-0.15, 0.15)
        s2 = rng.uniform(-0.15, 0.15)
        s3 = rng.uniform(-0.15, 0.15)

        starts.append([cx, cy, r1, r2, r3, a1, a2, a3, p1, p2, p3, q1, q2, q3, s1, s2, s3])
    return starts


def _powell_refine(params, time_limit):
    """Refine parameters using Powell's method."""
    start_time = time.time()

    def objective(x):
        return _score_params_neg(x)

    try:
        result = minimize(
            objective,
            params,
            method='Powell',
            options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12, 'disp': False}
        )
        new_params = result.x
        new_score = _score_params(new_params)
    except Exception:
        new_params = params
        new_score = _score_params(params)

    return new_params, new_score


def _nelder_mead_refine(params, time_limit):
    """Refine parameters using Nelder-Mead."""
    start_time = time.time()

    def objective(x):
        return _score_params_neg(x)

    try:
        result = minimize(
            objective,
            params,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True}
        )
        new_params = result.x
        new_score = _score_params(new_params)
    except Exception:
        new_params = params
        new_score = _score_params(params)

    return new_params, new_score


def _free_point_refine(pts, time_limit):
    """
    Refine all 26 coordinates freely using Powell, then Nelder-Mead.
    This allows the ring structure to break if beneficial.
    """
    start_time = time.time()
    x0 = pts.flatten()

    def objective(x):
        p = x.reshape(N, 2)
        return -_score(p)

    # Powell on free coordinates
    try:
        result = minimize(
            objective,
            x0,
            method='Powell',
            options={'maxiter': 300, 'ftol': 1e-12, 'xtol': 1e-12, 'disp': False}
        )
        new_pts = result.x.reshape(N, 2)
    except Exception:
        new_pts = pts.copy()

    # Nelder-Mead polish
    if time.time() - start_time < time_limit * 0.7:
        try:
            result2 = minimize(
                objective,
                new_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True}
            )
            new_pts = result2.x.reshape(N, 2)
        except Exception:
            pass

    return new_pts


def construct():
    """
    Main entry point: returns array of shape (N, 2).
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - 5  # 5 second safety margin

    rng = random.Random(42)

    # Generate diverse ring starts
    n_starts = 30
    starts = _generate_ring_starts(n_starts, rng)

    best_score = 0.0
    best_pts = None

    # Phase 1: Powell refinement on ring parameters
    for i, s in enumerate(starts):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 2:
            break

        # Quick score
        s_score = _score_params(s)
        if s_score > best_score:
            best_score = s_score
            best_pts = _params_to_points(s)

        # Refine with Powell (time-limited per start)
        t_per = max(0.5, min(3.0, remaining / max(1, n_starts - i)))
        refined_params, refined_score = _powell_refine(s, t_per)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = _params_to_points(refined_params)

    # Phase 2: Nelder-Mead on the best ring parameters
    if time.time() < deadline and best_pts is not None:
        # Reconstruct approximate params from best_pts (use center of mass)
        com = np.mean(best_pts, axis=0)
        # Estimate radii
        dists = np.linalg.norm(best_pts - com, axis=1)
        r_est = np.percentile(dists, [33, 66, 95])
        # Use a simple param vector
        approx_params = [com[0], com[1], r_est[0], r_est[1], r_est[2],
                         0, 0, 0, 2.0, 2.0, 2.0, 0, 0, 0, 0, 0, 0]
        nm_params, nm_score = _nelder_mead_refine(approx_params, time_limit=10)
        if nm_score > best_score:
            best_score = nm_score
            best_pts = _params_to_points(nm_params)

    # Phase 3: Free-coordinate refinement (break ring constraint)
    if time.time() < deadline and best_pts is not None:
        remaining = deadline - time.time()
        if remaining > 5:
            free_pts = _free_point_refine(best_pts, time_limit=remaining * 0.8)
            free_score = _score(free_pts)
            if free_score > best_score:
                best_score = free_score
                best_pts = free_pts

    # Phase 4: A few random free-coordinate perturbations + refinement
    if time.time() < deadline and best_pts is not None:
        remaining = deadline - time.time()
        n_perturb = 5
        for i in range(n_perturb):
            if time.time() > deadline:
                break
            if remaining < 3:
                break
            # Small random perturbation
            perturbed = best_pts + rng.gauss(0, 0.02, (N, 2))
            perturbed = np.clip(perturbed, 0.01, 0.99)
            p_score = _score(perturbed)
            if p_score > best_score:
                best_score = p_score
                best_pts = perturbed

            # Quick Powell on free coords
            t_per = max(0.3, min(2.0, (deadline - time.time()) / max(1, n_perturb - i)))
            refined = _free_point_refine(perturbed, time_limit=t_per)
            r_score = _score(refined)
            if r_score > best_score:
                best_score = r_score
                best_pts = refined

    # Phase 5: Differential evolution on ring parameters (global)
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 15:
            bounds = [
                (0.2, 0.8),  # cx
                (0.2, 0.8),  # cy
                (0.1, 0.35), # r1
                (0.25, 0.45),# r2
                (0.35, 0.55),# r3
                (0, 2*np.pi),# a1
                (0, 2*np.pi),# a2
                (0, 2*np.pi),# a3
                (0.5, 4.0),  # p1
                (0.5, 4.0),  # p2
                (0.5, 4.0),  # p3
                (-0.15, 0.15),# q1
                (-0.15, 0.15),# q2
                (-0.15, 0.15),# q3
                (-0.2, 0.2), # s1
                (-0.2, 0.2), # s2
                (-0.2, 0.2), # s3
            ]
            try:
                de_result = differential_evolution(
                    _score_params_neg,
                    bounds,
                    maxiter=50,
                    popsize=12,
                    tol=1e-8,
                    seed=42,
                    polish=False,
                    init='sobol',
                )
                de_params = de_result.x
                de_score = _score_params(de_params)
                if de_score > best_score:
                    best_score = de_score
                    best_pts = _params_to_points(de_params)

                # Powell polish on DE result
                if time.time() < deadline:
                    pp_params, pp_score = _powell_refine(de_params, time_limit=5)
                    if pp_score > best_score:
                        best_score = pp_score
                        best_pts = _params_to_points(pp_params)
            except Exception:
                pass

    # Final: ensure valid and in bounds
    if best_pts is None:
        # Fallback: regular hexagonal-ish arrangement
        best_pts = np.array([[0.5, 0.5]] +
                            [[0.5 + 0.3*np.cos(2*np.pi*k/6), 0.5 + 0.3*np.sin(2*np.pi*k/6)] for k in range(6)] +
                            [[0.5 + 0.45*np.cos(2*np.pi*k/6 + np.pi/6), 0.5 + 0.45*np.sin(2*np.pi*k/6 + np.pi/6)] for k in range(6)])
        best_pts = best_pts[:N]

    best_pts = np.clip(best_pts, 0, 1)

    # Final score check
    final_score = _score(best_pts)
    if final_score <= 0:
        # Fallback to a known-good configuration
        best_pts = np.array([
            [0.5, 0.5],
            [0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7],
            [0.2, 0.5], [0.8, 0.5], [0.5, 0.2], [0.5, 0.8],
            [0.35, 0.65], [0.65, 0.65], [0.65, 0.35], [0.35, 0.35],
        ])
        best_pts = np.clip(best_pts, 0, 1)

    return best_pts
# EVOLVE-BLOCK-END
