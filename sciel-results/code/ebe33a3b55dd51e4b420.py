"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import time

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _fourier_to_points(params):
    """
    params: [a1, b1, a2, b2, r0]
    a1, b1: 1st harmonic amplitudes
    a2, b2: 2nd harmonic amplitudes
    r0: base radius
    Returns 13 points.
    """
    a1, b1, a2, b2, r0 = params
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = r0 + a1 * np.cos(angles) + b1 * np.sin(angles) + a2 * np.cos(2 * angles) + b2 * np.sin(2 * angles)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack([x, y])
    # Center
    points = points - np.mean(points, axis=0)
    return points


def _fourier_objective(params):
    points = _fourier_to_points(params)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _smooth_objective(x_flat, p=50):
    """Smooth surrogate using log-sum-exp based soft-min for better gradients."""
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid

    if np.allclose(points, 0):
        return 1e10

    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10

    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)

    # Log-sum-exp based soft-min
    neg_p_areas = -p * areas
    max_neg = np.max(neg_p_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_p_areas - max_neg)))
    soft_min = -log_sum / p

    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score


def _objective_full(x_flat):
    """Full objective: minimize negative score."""
    points = x_flat.reshape(N, 2)
    points = points - np.mean(points, axis=0)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score


def _coordinate_descent_polish(points, deadline, max_iter=2000):
    """Coordinate descent on the hard objective: move each point's x and y independently."""
    best_pts = points.copy()
    best_s = _score(best_pts)

    for it in range(max_iter):
        if time.time() > deadline:
            break

        improved = False
        for i in range(N):
            for coord in range(2):
                # Try small perturbations in this coordinate
                for delta in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001]:
                    trial = best_pts.copy()
                    trial[i, coord] += delta
                    trial_s = _score(trial)
                    if trial_s > best_s + 1e-14:
                        best_pts = trial
                        best_s = trial_s
                        improved = True
                        break

        if not improved:
            break

    return best_pts, best_s


def _local_polish(points, deadline):
    """Multi-phase local optimization from a starting point."""
    best_x = points.flatten().copy()
    best_s = _score(points)

    # Phase 1: Powell on smooth objective (log-sum-exp based)
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=80),
                best_x,
                method='Powell',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead on full objective
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.x.copy()
        except Exception:
            pass

    # Phase 3: Coordinate descent polish on hard objective
    if time.time() < deadline - 2:
        try:
            cd_pts, cd_s = _coordinate_descent_polish(best_x.reshape(N, 2), deadline)
            if cd_s > best_s:
                best_s = cd_s
                best_x = cd_pts.flatten().copy()
        except Exception:
            pass

    return best_x.reshape(N, 2), best_s


def construct():
    """Construct 13 points to maximize the minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin

    best_points = None
    best_score = 0.0

    # Strategy 1: Fourier parameterization (low dimensionality, enforces smooth convexity)
    # params: [a1, b1, a2, b2, r0]
    # a1, b1, a2, b2 in [-0.2, 0.2], r0 in [0.8, 1.2]
    fourier_bounds = [(-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), (0.8, 1.2)]

    # Try a few random starts for Fourier optimization
    for seed in [42, 123, 456]:
        if time.time() > deadline - 20:
            break

        rng = np.random.RandomState(seed)
        x0 = np.array([rng.uniform(b[0], b[1]) for b in fourier_bounds])

        try:
            # Use Nelder-Mead for the 5D Fourier space
            res = minimize(
                _fourier_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if time.time() > deadline - 15:
                break

            pts = _fourier_to_points(res.x)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()

            # Polish the result in full 2D space
            if time.time() < deadline - 5:
                polished_pts, polished_s = _local_polish(pts, deadline)
                if polished_s > best_score:
                    best_score = polished_s
                    best_points = polished_pts.copy()

        except Exception:
            pass

    # If Fourier didn't find anything good, fall back to full space optimization
    if best_score < 1.0:
        # Start with a reasonable initial guess (regular polygon)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()

        # Differential evolution in full space
        bounds = [(-1, 1)] * (2 * N)
        try:
            result = differential_evolution(
                _objective_full,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-5,
                polish=False,
                init='sobol'
            )
            if time.time() > deadline - 10:
                pass
            else:
                candidate_points = result.x.reshape((-1, 2))
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()

                # Polish
                if time.time() < deadline - 5:
                    polished_pts, polished_s = _local_polish(candidate_points, deadline)
                    if polished_s > best_score:
                        best_score = polished_s
                        best_points = polished_pts.copy()
        except Exception:
            pass

    # Final polish if we have a good candidate
    if best_points is not None and time.time() < deadline - 2:
        try:
            final_pts, final_s = _local_polish(best_points, deadline)
            if final_s > best_score:
                best_score = final_s
                best_points = final_pts
        except Exception:
            pass

    if best_points is None:
        # Absolute fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points
# EVOLVE-BLOCK-END
