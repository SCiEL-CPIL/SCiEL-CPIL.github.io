"""Heilbronn triangles: place 11 points in a unit-area equilateral triangle
to maximize the smallest triangle area formed by any three points."""

import time
import numpy as np
from itertools import combinations

N = 11
# Vertices of the equilateral triangle (unit area = sqrt(3)/4)
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3) / 2])
AREA = np.sqrt(3) / 4.0

DEADLINE = time.time() + 130.0  # safety margin before 150s kill


def _point_in_triangle(p):
    """Check if point p is inside or on the triangle (vectorized for arrays)."""
    # Barycentric coordinates
    v0 = V2 - V1
    v1 = V3 - V1
    v2 = p - V1
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    den = d00 * d11 - d01 * d01
    u = (d11 * d20 - d01 * d21) / den
    v = (d00 * d21 - d01 * d20) / den
    return (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)


def _project_to_triangle(p):
    """Project point p into the triangle if outside."""
    # Use barycentric coords, clip
    v0 = V2 - V1
    v1 = V3 - V1
    v2 = p - V1
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    den = d00 * d11 - d01 * d01
    u = (d11 * d20 - d01 * d21) / den
    v = (d00 * d21 - d01 * d20) / den
    # Clip to [0,1] with u+v<=1
    if u < 0:
        u = 0
    if v < 0:
        v = 0
    if u + v > 1:
        s = u + v
        u /= s
        v /= s
    return V1 + u * v0 + v * v1


def _min_area(pts):
    """Minimum triangle area among all triples of points."""
    n = len(pts)
    min_a = np.inf
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) -
                                 (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_a:
                    min_a = area
    return min_a


def _objective(pts):
    """Negative minimum area (to minimize)."""
    return -_min_area(pts)


def _nelder_mead(x0, max_iter=200, tol=1e-8):
    """Simple Nelder-Mead on flat array of 2*N coords."""
    n = len(x0)
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        step = x0.copy()
        step[i] += 0.05 if step[i] < 1 else -0.05
        simplex.append(step)
    simplex = np.array(simplex)
    # Evaluate
    fvals = np.array([_objective(np.array(simplex[i]).reshape(N, 2)) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        # Sort
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if fvals[-1] - fvals[0] < tol:
            break
        
        centroid = simplex[:-1].mean(axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = _objective(np.array(xr).reshape(N, 2))
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = _objective(np.array(xe).reshape(N, 2))
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = _objective(np.array(xc).reshape(N, 2))
                if fc <= fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = _objective(np.array(simplex[i]).reshape(N, 2))
                    break
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = _objective(np.array(simplex[i]).reshape(N, 2))
                break
    
    return simplex[0], fvals[0]


def _grid_init(seed):
    """Generate a structured grid-based initial configuration."""
    rng = np.random.RandomState(seed)
    pts = []
    # Use a 4x4 grid in barycentric coords, pick 11 with jitter
    grid = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if i + j + k == 3:
                    grid.append([i / 3.0, j / 3.0, k / 3.0])
    # Shuffle and pick 11
    idx = rng.permutation(len(grid))[:N]
    for i in idx:
        b, c, a = grid[i]  # barycentric: a*V1 + b*V2 + c*V3
        p = b * V2 + c * V3
        # Add small jitter
        p = p + rng.uniform(-0.02, 0.02, 2)
        p = _project_to_triangle(p)
        pts.append(p)
    return np.array(pts)


def _random_init(seed):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < N:
        u = rng.random()
        v = rng.random()
        if u + v < 1.0:
            p = u * V2 + v * V3
            pts.append(p)
    return np.array(pts)


def _structured_init(seed):
    """Structured initialization based on known good patterns."""
    rng = np.random.RandomState(seed)
    # Start with vertices, centroid, edge midpoints, and a few interior
    base = [
        V1 * 0.05 + V2 * 0.05 + V3 * 0.05,  # near V1
        V1 * 0.05 + V2 * 0.95 + V3 * 0.05,  # near V2
        V1 * 0.05 + V2 * 0.05 + V3 * 0.95,  # near V3
        V1 * 0.33 + V2 * 0.33 + V3 * 0.33,  # centroid
        V1 * 0.5 + V2 * 0.5 + V3 * 0.0,     # mid V1-V2
        V1 * 0.5 + V2 * 0.0 + V3 * 0.5,     # mid V1-V3
        V1 * 0.0 + V2 * 0.5 + V3 * 0.5,     # mid V2-V3
        V1 * 0.25 + V2 * 0.25 + V3 * 0.5,   # interior
        V1 * 0.5 + V2 * 0.25 + V3 * 0.25,   # interior
        V1 * 0.25 + V2 * 0.5 + V3 * 0.25,   # interior
        V1 * 0.15 + V2 * 0.15 + V3 * 0.7,   # interior
    ]
    # Add jitter
    pts = []
    for p in base:
        p = p + rng.uniform(-0.03, 0.03, 2)
        p = _project_to_triangle(p)
        pts.append(p)
    return np.array(pts)


def construct():
    best_pts = None
    best_val = -np.inf
    
    # Multi-start with different initializations
    inits = []
    for seed in range(5):
        inits.append(_grid_init(seed))
    for seed in range(5):
        inits.append(_random_init(seed))
    for seed in range(3):
        inits.append(_structured_init(seed))
    
    for init in inits:
        if time.time() > DEADLINE - 10:
            break
        flat = init.flatten()
        opt_flat, fval = _nelder_mead(flat, max_iter=150)
        opt_pts = opt_flat.reshape(N, 2)
        # Ensure all points are in triangle
        valid = _point_in_triangle(opt_pts)
        if not np.all(valid):
            for i in range(N):
                if not valid[i]:
                    opt_pts[i] = _project_to_triangle(opt_pts[i])
        val = _min_area(opt_pts)
        if val > best_val:
            best_val = val
            best_pts = opt_pts.copy()
    
    if best_pts is None:
        # Fallback
        best_pts = _random_init(0)
    
    return best_pts
