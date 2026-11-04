"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    min_area = float('inf')
    for i in range(N):
        pi = points[i]
        for j in range(i + 1, N):
            pj = points[j]
            for k in range(j + 1, N):
                pk = points[k]
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                v1 = pj - pi
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10


def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _objective(x):
    """Objective function for minimizers (minimizes negative score)."""
    points = x.reshape(-1, 2)
    return -_score(points)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Bounds for coordinates
    bounds = [(-1, 1)] * (2 * N)

    # 1. Global Search: Differential Evolution
    # Robust for non-smooth, non-convex problems
    try:
        res_de = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-5,
            polish=False  # Disable built-in polish to use our own better one
        )
        best_x = res_de.x
        best_score = -res_de.fun
    except Exception:
        # Fallback to random if DE fails
        best_x = np.random.uniform(-1, 1, size=2 * N)
        best_score = -_objective(best_x)

    if time.time() > deadline:
        return best_x.reshape(-1, 2)

    # 2. Local Refinement: COBYLA
    # COBYLA is derivative-free and handles non-smooth functions well
    # We try a few restarts from the best global solution with small perturbations
    best_x_final = best_x
    best_score_final = best_score

    rng = np.random.RandomState(42)
    for i in range(10):  # 10 local refinement attempts
        if time.time() > deadline:
            break

        # Perturb the best solution
        # Smaller perturbations as we go on
        scale = 0.1 / (i + 1)
        x0 = best_x + rng.normal(0, scale, size=best_x.shape)

        try:
            res_local = minimize(
                _objective,
                x0,
                method='COBYLA',
                options={'maxiter': 200, 'rhobeg': 0.05}
            )

            if res_local.fun < -best_score_final:
                best_score_final = -res_local.fun
                best_x_final = res_local.x
        except Exception:
            continue

    return best_x_final.reshape(-1, 2)
# EVOLVE-BLOCK-END
