"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) triangle index triplets
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
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas.min()


def _is_inside(p):
    """Check if point p=(x,y) is inside the equilateral triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, H)
    # Barycentric check
    x, y = p
    # Must be above y=0
    if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9:
        return False
    # Left edge: from (0,0) to (0.5, H): y <= sqrt(3)*x
    if y > np.sqrt(3) * x + 1e-9:
        return False
    # Right edge: from (1,0) to (0.5, H): y <= sqrt(3)*(1-x)
    if y > np.sqrt(3) * (1 - x) + 1e-9:
        return False
    return True


def _params_to_points(params):
    """Convert 2*N params to N points inside triangle using barycentric coords.
    params[i] = (u, v) where u,v in [0,1], u+v<=1 -> point = u*B + v*C + (1-u-v)*A
    """
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[2 * i]
        v = params[2 * i + 1]
        # Ensure u+v <= 1 by clamping
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        points[i] = np.array([u * 1.0 + v * 0.5, v * H])
    return points


def _objective(params):
    """Minimize negative min-area (so we maximize min-area)."""
    points = _params_to_points(params)
    # Check all points are inside (should be by construction, but safety)
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6  # Large penalty
    return -_min_area(points)


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill

    best_params = None
    best_score = -np.inf

    # Phase 1: Differential Evolution for global search
    bounds = [(0.0, 1.0)] * (2 * N)
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=False,
            updating='immediate',
            init='sobol'
        )
        if time.time() > deadline:
            # Return best found so far
            if best_params is not None:
                return _params_to_points(best_params)
            # Fallback: random
            rng = np.random.RandomState(42)
            pts = []
            for _ in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        pts.append([u + 0.5 * v, H * v])
                        break
            return np.array(pts)
        best_params = result.x
        best_score = -result.fun
    except Exception:
        pass

    # Phase 2: Local polish with Nelder-Mead (more robust than gradient methods here)
    if time.time() < deadline - 20.0:
        try:
            res2 = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if -res2.fun > best_score:
                best_params = res2.x
                best_score = -res2.fun
        except Exception:
            pass

    # Phase 3: Try a few random restarts with local polish if time remains
    if time.time() < deadline - 10.0:
        rng = np.random.RandomState(123)
        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
            # Random initialization in barycentric coords
            params = np.zeros(2 * N)
            for i in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        params[2 * i] = u
                        params[2 * i + 1] = v
                        break
            try:
                res = minimize(
                    _objective,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if -res.fun > best_score:
                    best_params = res.x
                    best_score = -res.fun
            except Exception:
                pass

    # Final check
    if time.time() > deadline:
        pass  # Just return what we have

    if best_params is not None:
        points = _params_to_points(best_params)
        # Final safety: ensure all points are strictly inside
        for i in range(N):
            if not _is_inside(points[i]):
                # Nudge inward
                points[i] = 0.95 * points[i] + 0.05 * np.array([0.5, H / 3])
        return points
    else:
        # Absolute fallback
        rng = np.random.RandomState(0)
        pts = []
        for _ in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 0.99:
                    pts.append([u + 0.5 * v, H * v])
                    break
        return np.array(pts)

# EVOLVE-BLOCK-END
