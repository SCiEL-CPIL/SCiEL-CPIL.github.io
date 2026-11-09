"""Heilbronn triangles: maximize the minimum triangle area for 11 points in a unit triangle."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools
import sys

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all triangle index triples
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
_n_tri = len(_idx_i)


def _bary_to_cart(bary):
    """Convert barycentric coordinates (N,3) to Cartesian (N,2)."""
    # bary[:,0] * A + bary[:,1] * B + bary[:,2] * C
    # A = (0,0), B = (1,0), C = (0.5, H)
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = H * bary[:, 2]
    return np.column_stack([x, y])


def _min_area(points):
    """Compute the minimum triangle area among all C(N,3) triangles."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * |(p2-p1) x (p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _neg_min_area_bary(bary_flat):
    """Objective for optimization: negative of min area, in barycentric coords."""
    bary = bary_flat.reshape(N, 3)
    # Normalize barycentric coordinates to sum to 1
    s = bary.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    bary = bary / s
    # Clip to [0,1]
    bary = np.clip(bary, 0.0, 1.0)
    points = _bary_to_cart(bary)
    return -_min_area(points)


def _neg_min_area_cart(pts_flat):
    """Objective for local refinement in Cartesian coordinates."""
    points = pts_flat.reshape(N, 2)
    return -_min_area(points)


def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin before 150s kill

    # --- Stage 1: Global search with differential evolution on barycentric coords ---
    # Parameters: N*3 barycentric coords, each in [0, 1]
    bounds = [(0.0, 1.0)] * (N * 3)

    best_result = None
    best_score = -np.inf

    # Run differential evolution with time-aware stopping
    def _time_check():
        return (time.time() - t0) > deadline - 30.0

    try:
        de_result = differential_evolution(
            _neg_min_area_bary,
            bounds=bounds,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            callback=lambda xk, convergence: convergence > 0.01,
            init='sobol',
            updating='immediate',
            maxfevals=5000
        )
        if de_result.success or de_result.fun < best_score:
            best_score = -de_result.fun
            best_result = de_result.x
    except Exception:
        pass

    if (time.time() - t0) > deadline - 25.0:
        # Not enough time for refinement, return what we have
        if best_result is not None:
            bary = best_result.reshape(N, 3)
            s = bary.sum(axis=1, keepdims=True)
            s = np.where(s < 1e-12, 1.0, s)
            bary = bary / s
            bary = np.clip(bary, 0.0, 1.0)
            return _bary_to_cart(bary)
        # Fallback: return a simple grid-like arrangement
        return _fallback_points()

    # --- Stage 2: Local refinement in Cartesian coordinates ---
    if best_result is not None:
        bary = best_result.reshape(N, 3)
        s = bary.sum(axis=1, keepdims=True)
        s = np.where(s < 1e-12, 1.0, s)
        bary = bary / s
        bary = np.clip(bary, 0.0, 1.0)
        cart = _bary_to_cart(bary).flatten()
    else:
        cart = _fallback_points().flatten()

    # Local refinement using L-BFGS-B
    try:
        res = minimize(
            _neg_min_area_cart,
            x0=cart,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.fun < best_score:
            best_score = -res.fun
            best_result_cart = res.x
        else:
            best_result_cart = cart
    except Exception:
        best_result_cart = cart

    points = best_result_cart.reshape(N, 2)

    # Ensure points are inside the triangle (clip if needed)
    points = _clip_to_triangle(points)

    return points


def _clip_to_triangle(points):
    """Clip points to be inside the triangle."""
    pts = points.copy()
    # Check against three edges
    # Edge AB: y >= 0
    pts[:, 1] = np.clip(pts[:, 1], 0.0, H)
    # Edge BC: y <= -H*(x - 1) => y + H*x <= H
    # Edge CA: y <= H*x => y - H*x <= 0
    # For each point, check and project if outside
    for i in range(N):
        x, y = pts[i]
        # Check y >= 0
        if y < 0:
            y = 0
        # Check y <= H
        if y > H:
            y = H
        # Check y <= H * x (edge CA)
        if y > H * x:
            # Project onto y = H*x
            # Minimize (x'-x)^2 + (y'-y)^2 subject to y' = H*x'
            # x' = (x + H*y) / (1 + H^2)
            x = (x + H * y) / (1 + H * H)
            y = H * x
        # Check y <= H*(1-x) (edge BC)
        if y > H * (1 - x):
            # Project onto y = H*(1-x) => y + H*x = H
            # Minimize (x'-x)^2 + (y'-y)^2 subject to H*x' + y' = H
            # x' = (H*x + y + H*(H - H*x - y)) / (H^2 + 1) ... let me redo
            # Constraint: H*x' + y' = H
            # Lagrange: 2(x'-x) + lambda*H = 0, 2(y'-y) + lambda = 0
            # lambda = -2(y'-y), so x' - x = H*(y'-y)
            # x' = x + H*(y'-y)
            # Substitute: H*(x + H*(y'-y)) + y' = H
            # H*x + H^2*y' - H^2*y + y' = H
            # y'*(H^2 + 1) = H - H*x + H^2*y
            # y' = (H - H*x + H^2*y) / (H^2 + 1)
            y = (H - H * x + H * H * y) / (H * H + 1)
            x = x + H * (y - y)  # This is wrong, let me redo
            # Actually from x' = x + H*(y'-y), and we just found y'
            x = x + H * (y - y)  # no, y was updated
            # Let me just recompute properly
            y_new = (H - H * x + H * H * y) / (H * H + 1)
            x_new = x + H * (y_new - y)
            x, y = x_new, y_new

        pts[i] = [x, y]
    return pts


def _fallback_points():
    """Return a simple arrangement of 11 points as fallback."""
    # Use a 3x3 grid plus center and one extra
    pts = []
    for i in range(3):
        for j in range(3):
            x = 0.2 + 0.3 * i
            y = 0.2 * H + 0.3 * H * j
            pts.append([x, y])
    pts.append([0.5, H / 2])
    pts.append([0.5, H / 4])
    return np.array(pts[:N])


# Quick test
if __name__ == '__main__':
    pts = construct()
    print("Points shape:", pts.shape)
    print("Min area:", _min_area(pts))
    print("Triangle area:", 0.5 * 1.0 * H)
    print("Ratio:", _min_area(pts) / (0.5 * H))
