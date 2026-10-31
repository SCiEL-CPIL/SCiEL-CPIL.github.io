import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import sys

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        # Check for degeneracy
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / hull area."""
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


def _negative_score(x_flat):
    """Objective for optimization: -score."""
    points = x_flat.reshape(N, 2)
    # Check for NaNs or Infs
    if not np.all(np.isfinite(points)):
        return 1e10
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin against 120s limit
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        if not np.all(np.isfinite(points)):
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Phase 1: Initial Guesses
    # 1. Regular 13-gon
    _update_best(_regular_polygon(N, 1.0))
    
    # 2. Concentric Rings: 1 center, 6 inner, 6 outer
    center = np.array([[0.0, 0.0]])
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    inner_pts = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    outer_angles = np.linspace(np.pi/6, 2 * np.pi + np.pi/6, 6, endpoint=False)
    outer_pts = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
    _update_best(np.vstack([center, inner_pts, outer_pts]))
    
    # 3. Random points
    rng = np.random.default_rng(42)
    for _ in range(10):
        pts = rng.uniform(-1, 1, (N, 2))
        _update_best(pts)

    # Phase 2: Differential Evolution
    # DE is a global optimizer, good for non-smooth functions.
    # It requires more function evaluations, but is robust.
    bounds = [(-1.5, 1.5)] * (N * 2)
    
    # We need to be careful with the time limit.
    # DE with popsize=20, maxiter=50 will do ~1000 evaluations.
    # Each evaluation is ~1ms? 1000ms = 1s.
    # We can afford more.
    
    try:
        # Use a wrapper to enforce time limit if DE runs long?
        # scipy DE doesn't have a direct time limit, but we can check in the objective?
        # No, better to just set maxiter conservatively.
        
        result = differential_evolution(
            _negative_score,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-4,
            init='sobol' # Better initialization
        )
        
        opt_points = result.x.reshape(N, 2)
        _update_best(opt_points)
        
    except Exception as e:
        print(f"DE failed: {e}")

    # Phase 3: Local Polish with Nelder-Mead
    if best_points is not None and time.time() < deadline - 10:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _negative_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            opt_points = result.x.reshape(N, 2)
            _update_best(opt_points)
        except Exception as e:
            print(f"NM failed: {e}")

    # Phase 4: Additional Random Restarts with DE if time permits
    if time.time() < deadline - 20:
        for seed in [123, 456]:
            if time.time() > deadline - 10:
                break
            try:
                result = differential_evolution(
                    _negative_score,
                    bounds=bounds,
                    seed=seed,
                    maxiter=20,
                    popsize=15,
                    tol=1e-3,
                    init='latinhypercube'
                )
                opt_points = result.x.reshape(N, 2)
                _update_best(opt_points)
                
                # Quick polish
                if time.time() < deadline - 5:
                    result2 = minimize(
                        _negative_score,
                        opt_points.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9}
                    )
                    _update_best(result2.x.reshape(N, 2))
            except Exception:
                pass

    # Final check
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
    
    return best_points
