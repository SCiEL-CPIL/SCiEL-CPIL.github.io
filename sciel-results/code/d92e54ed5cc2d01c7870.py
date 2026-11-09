"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2

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
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.min(np.abs(cross))


def _is_inside(points):
    """Check if all points are inside the equilateral triangle (0,0), (1,0), (0.5, H)."""
    x = points[:, 0]
    y = points[:, 1]
    # Inside triangle: y >= 0, y <= H, and y <= sqrt(3)*x for left edge, y <= sqrt(3)*(1-x) for right edge
    # More precisely: 0 <= y, y <= H, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    # Since the triangle has vertices (0,0), (1,0), (0.5, H):
    # Left edge from (0,0) to (0.5, H): y = sqrt(3)*x
    # Right edge from (1,0) to (0.5, H): y = -sqrt(3)*(x-1) = sqrt(3)*(1-x)
    # Bottom edge: y = 0
    cond1 = y >= 0
    cond2 = y <= H
    cond3 = y <= np.sqrt(3) * x
    cond4 = y <= np.sqrt(3) * (1 - x)
    return np.all(cond1 & cond2 & cond3 & cond4)


def _params_to_points(params):
    """Convert 2N parameters to points, ensuring they're inside the triangle."""
    x = params[:N]
    y = params[N:]
    points = np.column_stack([x, y])
    # Clip to be inside triangle
    # For each point, if outside, project onto nearest edge
    for i in range(N):
        px, py = points[i]
        if py < 0:
            py = 0
        if py > H:
            py = H
        if py > np.sqrt(3) * px:
            # Project onto left edge y = sqrt(3)*x
            # Closest point on line y=sqrt(3)*x to (px,py)
            # Line: sqrt(3)*x - y = 0
            d = (np.sqrt(3) * px - py) / (3 + 1)  # denominator is 3+1=4
            px -= np.sqrt(3) * d
            py -= 1 * d
        if py > np.sqrt(3) * (1 - px):
            # Project onto right edge y = sqrt(3)*(1-x)
            # Line: sqrt(3)*x + y - sqrt(3) = 0
            d = (np.sqrt(3) * px + py - np.sqrt(3)) / (3 + 1)
            px -= np.sqrt(3) * d
            py -= 1 * d
        points[i] = [px, py]
    return points


def _objective(params):
    """Objective function to minimize: negative of min area."""
    points = _params_to_points(params)
    if not _is_inside(points):
        return 1e6
    return -_min_area(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_points = None
    best_score = -np.inf

    # Strategy 1: Differential evolution with multiple restarts
    bounds = [(0, 1)] * N + [(0, H)] * N

    # Run differential evolution
    de_result = differential_evolution(
        _objective,
        bounds=bounds,
        maxiter=200,
        popsize=30,
        tol=1e-10,
        seed=42,
        polish=False,
        init='sobol'
    )
    
    if time.time() > deadline:
        # Time's up, return best found or a default
        if best_points is None:
            # Generate a reasonable default
            pts = []
            for i in range(N):
                u = (i + 1) / (N + 1)
                v = 0.3
                if u + v < 0.95:
                    pts.append([u + 0.5 * v, H * v])
                else:
                    pts.append([0.5, 0.3 * H])
            best_points = np.array(pts)
        return best_points

    de_points = _params_to_points(de_result.x)
    de_score = _min_area(de_points)
    if de_score > best_score:
        best_score = de_score
        best_points = de_points.copy()

    # Strategy 2: Local refinement using Nelder-Mead
    if time.time() < deadline - 10:
        # Try to refine with Nelder-Mead
        res = minimize(
            _objective,
            de_result.x,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        nm_points = _params_to_points(res.x)
        nm_score = _min_area(nm_points)
        if nm_score > best_score:
            best_score = nm_score
            best_points = nm_points.copy()

    # Strategy 3: Try a few more random restarts with local refinement
    rng = np.random.RandomState(123)
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        x0 = rng.uniform(0, 1, size=2 * N)
        x0[N:] *= H
        res = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        pts = _params_to_points(res.x)
        sc = _min_area(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()

    return best_points
