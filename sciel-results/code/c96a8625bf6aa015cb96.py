"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2  # area of equilateral triangle with side 1 = sqrt(3)/4
DEADLINE = 145  # safety margin from 150s limit


def _bary_to_cart(u, v):
    """Convert barycentric-like coords (u,v) in [0,1] with u+v<=1 to Cartesian."""
    return np.column_stack([u + 0.5 * v, S3H * v])


def _cart_to_bary(pts):
    """Convert Cartesian points to (u,v) where u+v<=1, u>=0, v>=0."""
    v = pts[:, 1] / S3H
    u = pts[:, 0] - 0.5 * v
    return np.column_stack([u, v])


def _project_to_triangle(pts):
    """Project points into the triangle using barycentric coords."""
    uv = _cart_to_bary(pts)
    # Clip to valid barycentric region: u>=0, v>=0, u+v<=1
    uv[:, 0] = np.clip(uv[:, 0], 0, 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, 1)
    # Fix u+v>1
    s = uv[:, 0] + uv[:, 1]
    mask = s > 1
    if np.any(mask):
        uv[mask, 0] /= s[mask]
        uv[mask, 1] /= s[mask]
    return _bary_to_cart(uv)


def _min_area(pts):
    """Compute minimum triangle area among all triples, normalized by triangle area."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                if cross < min_a:
                    min_a = cross
    return min_a / TRI_AREA


def _negative_min_area(x):
    """Objective: negative of min area (for minimization). x is 2N barycentric coords."""
    pts = _bary_to_cart(x.reshape(N, 2))
    pts = _project_to_triangle(pts)
    return -_min_area(pts)


def _penalized_objective(x):
    """Penalized objective for L-BFGS-B: penalize if points leave triangle."""
    uv = x.reshape(N, 2)
    # Penalty for being outside barycentric triangle
    penalty = 0.0
    for i in range(N):
        u, v = uv[i]
        if u < 0: penalty += 100 * u * u
        if v < 0: penalty += 100 * v * v
        if u + v > 1: penalty += 100 * (u + v - 1) * (u + v - 1)
    pts = _bary_to_cart(uv)
    return -_min_area(pts) + penalty


def _local_optimize(x0, time_limit):
    """Run L-BFGS-B then Nelder-Mead on a starting point. Returns best pts and score."""
    best_score = -_negative_min_area(x0)
    best_x = x0.copy()
    
    t_start = time.time()
    
    # L-BFGS-B
    try:
        res = minimize(_penalized_objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if time.time() - t_start > time_limit * 0.7:
            return _project_to_triangle(_bary_to_cart(best_x)), best_score
        if res.fun < _negative_min_area(best_x):
            best_x = res.x
            best_score = -res.fun
    except Exception:
        pass
    
    if time.time() - t_start > time_limit * 0.85:
        return _project_to_triangle(_bary_to_cart(best_x)), best_score
    
    # Nelder-Mead polish
    try:
        res2 = minimize(_negative_min_area, best_x, method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < _negative_min_area(best_x):
            best_x = res2.x
            best_score = -res2.fun
    except Exception:
        pass
    
    return _project_to_triangle(_bary_to_cart(best_x)), best_score


def construct():
    t0 = time.time()
    remaining = DEADLINE - (t0 - t0)  # full budget
    
    # Initial diverse starting points
    np.random.seed(42)
    
    best_score = -1
    best_pts = None
    
    # Generate initial candidates
    candidates = []
    
    # 1. Uniform random
    for trial in range(20):
        uv = np.random.rand(N, 2)
        uv[:, 0] += 0.5 * uv[:, 1]  # wait, need u+v<=1
        # Proper random in triangle
        u1 = np.random.rand(N)
        u2 = np.random.rand(N)
        m = np.maximum(u1, u2)
        mn = np.minimum(u1, u2)
        uv = np.column_stack([mn, m - mn])  # u+v = m <= 1
        candidates.append(uv.ravel())
    
    # 2. Symmetric grid-like patterns
    # Corners + center
    uv_grid = np.array([
        [0.01, 0.01], [0.98, 0.01], [0.01, 0.98],
        [0.33, 0.33], [0.5, 0.15], [0.15, 0.5],
        [0.6, 0.3], [0.3, 0.6], [0.45, 0.45],
        [0.2, 0.2], [0.7, 0.2]
    ])
    candidates.append(uv_grid.ravel())
    
    # 3. Fekete-like: try to space points evenly
    # Use a spiral or golden ratio distribution
    for start_angle in [0, np.pi/3, 2*np.pi/3]:
        uv_spiral = np.zeros((N, 2))
        for i in range(N):
            r = np.sqrt((i + 0.5) / N) * 0.95
            theta = start_angle + i * 2 * np.pi / N
            # Map to triangle: use barycentric
            u = r * np.cos(theta) * 0.5 + 0.5
            v = r * np.sin(theta) * 0.5 + 0.5
            # Ensure u+v<=1
            if u + v > 1:
                scale = 1.0 / (u + v)
                u *= scale
                v *= scale
            uv_spiral[i] = [u, v]
        candidates.append(uv_spiral.ravel())
    
    # 4. Differential evolution global search
    bounds = [(0, 1)] * (2 * N)
    
    def de_objective(x):
        return _negative_min_area(x)
    
    try:
        de_res = differential_evolution(de_objective, bounds, seed=42, maxiter=50,
                                         popsize=15, tol=1e-8, polish=False,
                                         update_interval=10)
        if de_res.fun < _negative_min_area(candidates[0]):
            candidates.insert(0, de_res.x)
    except Exception:
        pass
    
    if time.time() - t0 > DEADLINE - 30:
        # Not enough time for local opt, just return best from DE or random
        for c in candidates[:5]:
            pts = _project_to_triangle(_bary_to_cart(c))
            s = _min_area(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
        if best_pts is None:
            best_pts = _project_to_triangle(_bary_to_cart(candidates[0]))
            best_score = _min_area(best_pts)
        return best_pts
    
    # Local optimize each candidate
    time_per_candidate = min(10, (DEADLINE - 5 - (time.time() - t0)) / max(len(candidates), 1))
    
    for i, c in enumerate(candidates):
        if time.time() - t0 > DEADLINE - 10:
            break
        try:
            pts, score = _local_optimize(c, time_per_candidate)
            if score > best_score:
                best_score = score
                best_pts = pts
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback
        uv = np.random.rand(N, 2)
        u1 = np.random.rand(N)
        u2 = np.random.rand(N)
        m = np.maximum(u1, u2)
        mn = np.minimum(u1, u2)
        uv = np.column_stack([mn, m - mn])
        best_pts = _bary_to_cart(uv)
        best_score = _min_area(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
