"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)


def _is_inside(points, margin=0.005):
    """Check if all points are inside the triangle with a small margin."""
    # Barycentric: point = u*A + v*B + w*C, u+v+w=1, u,v,w >= 0
    # For triangle (0,0), (1,0), (0.5, H):
    # w = y / H
    # v = (x - 0.5*w) / 0.5 ... actually let's use the standard formula
    x = points[:, 0]
    y = points[:, 1]
    # Barycentric coordinates
    w = y / H
    v = (x - 0.5 * w) * 2  # since B=(1,0), C=(0.5,H)
    u = 1 - v - w
    # Actually let me re-derive: P = u*(0,0) + v*(1,0) + w*(0.5,H)
    # x = v + 0.5*w
    # y = H*w
    # w = y/H
    # v = x - 0.5*w
    # u = 1 - v - w
    v = x - 0.5 * w
    u = 1 - v - w
    return np.all(u >= -margin) and np.all(v >= -margin) and np.all(w >= -margin)


def _params_to_points(params):
    """Convert 2N parameters (barycentric coords) to Cartesian points."""
    # params: (2N,) where pairs are (v, w) for each point, u = 1-v-w
    v = params[0::2]
    w = params[1::2]
    u = 1 - v - w
    # P = v*B + w*C = v*(1,0) + w*(0.5, H)
    x = v + 0.5 * w
    y = H * w
    return np.column_stack([x, y])


def _objective(params):
    """Negative min area (to minimize)."""
    points = _params_to_points(params)
    if not _is_inside(points, margin=0.0):
        return 1.0  # large penalty
    return -_min_area(points)


def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # 10s safety margin

    # Bounds for barycentric coords v, w: each in [0, 1] but constrained by v+w <= 1
    # For DE, we use bounds [0,1] for each and rely on the penalty for invalid points
    # Better: use a tighter bound. Since v+w<=1, we can bound v in [0,1], w in [0,1-v]
    # But DE needs fixed bounds. Use [0, 1] for both and penalize.
    bounds = [(0.0, 1.0)] * (2 * N)

    # Initial population for DE
    pop_size = 80
    max_iter = 200

    def time_check():
        return time.time() < deadline

    # Run differential evolution with time callback
    best_result = None
    best_score = -np.inf

    # Use a callback to check time
    class TimeCallback:
        def __init__(self):
            self.stop = False
        def __call__(self, xk, fk):
            if time.time() > deadline - 15.0:
                self.stop = True
            return self.stop

    callback = TimeCallback()

    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=max_iter,
            popsize=pop_size,
            tol=1e-8,
            mutation=(0.4, 1.0),
            recombination=0.7,
            seed=42,
            polish=False,
            callback=callback,
            workers=1
        )
        if result.success or result.x is not None:
            best_params = result.x
            best_points = _params_to_points(best_params)
            if _is_inside(best_points, margin=-0.001):
                best_score = _min_area(best_points)
                best_result = best_points
    except Exception:
        pass

    # Local polish with Nelder-Mead
    if best_result is not None and time.time() < deadline - 20.0:
        def polish_objective(p):
            pts = _params_to_points(p)
            if not _is_inside(pts, margin=0.0):
                return 1.0
            return -_min_area(pts)

        try:
            res = minimize(
                polish_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            polish_points = _params_to_points(res.x)
            if _is_inside(polish_points, margin=-0.001):
                polish_score = _min_area(polish_points)
                if polish_score > best_score:
                    best_score = polish_score
                    best_result = polish_points
        except Exception:
            pass

    # If we still have time, try a few random restarts with local optimization
    if best_result is None:
        # Fallback: use the seed-like random points
        import random
        rng = random.Random(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, H * v])
        best_result = np.array(points)
        best_score = _min_area(best_result)
    elif time.time() < deadline - 30.0:
        # Try a few random restarts
        rng = np.random.RandomState(123)
        for restart in range(3):
            if time.time() > deadline - 20.0:
                break
            init_params = rng.uniform(0.05, 0.95, size=2 * N)
            # Ensure valid: v+w <= 0.95
            for i in range(N):
                if init_params[2*i] + init_params[2*i+1] > 0.95:
                    s = 0.95 / (init_params[2*i] + init_params[2*i+1])
                    init_params[2*i] *= s
                    init_params[2*i+1] *= s

            def ro(p):
                pts = _params_to_points(p)
                if not _is_inside(pts, margin=0.0):
                    return 1.0
                return -_min_area(pts)

            try:
                res = minimize(ro, init_params, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
                pts = _params_to_points(res.x)
                if _is_inside(pts, margin=-0.001):
                    sc = _min_area(pts)
                    if sc > best_score:
                        best_score = sc
                        best_result = pts
            except Exception:
                pass

    # Ensure we return a valid result
    if best_result is None:
        import random
        rng = random.Random(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, H * v])
        best_result = np.array(points)

    return best_result
