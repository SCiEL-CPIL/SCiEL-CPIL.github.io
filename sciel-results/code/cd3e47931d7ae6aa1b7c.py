"""Heilbronn triangles: place 11 points to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0

# Precompute all C(11,3) = 165 triangle index triples
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points.
    pts has shape (N, 2)."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross).min()


def _project_to_triangle(pts):
    """Project points to be inside the triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x = pts[:, 0]
    y = pts[:, 1]
    # Barycentric coordinates
    # Point = a*(0,0) + b*(1,0) + c*(0.5, sqrt(3)/2)
    # x = b + 0.5*c
    # y = sqrt(3)/2 * c
    # => c = 2*y/sqrt(3), b = x - 0.5*c, a = 1 - b - c
    c = 2.0 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1.0 - b - c
    # Clamp to [0,1] and renormalize
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0, 1)
    c = np.clip(c, 0, 1)
    s = a + b + c
    a /= s
    b /= s
    c /= s
    # Convert back
    x_new = b + 0.5 * c
    y_new = _H * c
    return np.column_stack([x_new, y_new])


def _objective(x):
    """Objective to minimize: negative of min triangle area."""
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_min_triangle_area(pts)


def _symmetric_initial_config():
    """Create a symmetric initial configuration for 11 points."""
    # Use a grid-like pattern with symmetry
    # Triangle vertices
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, _H])
    
    points = []
    
    # Place points in a structured way
    # Row 0 (bottom): 4 points
    for i in range(4):
        t = (i + 0.5) / 4.0
        points.append(t * v1)
    
    # Row 1: 3 points
    for i in range(3):
        t = (i + 0.5) / 3.0
        x = t * 1.0
        y = 0.33 * _H
        # Interpolate between edges
        frac = y / _H
        x_left = frac * 0.5
        x_right = 1.0 - frac * 0.5
        x = x_left + frac * (x_right - x_left) * t
        points.append(np.array([x, y]))
    
    # Row 2: 2 points
    for i in range(2):
        t = (i + 0.5) / 2.0
        y = 0.66 * _H
        frac = y / _H
        x_left = frac * 0.5
        x_right = 1.0 - frac * 0.5
        x = x_left + frac * (x_right - x_left) * t
        points.append(np.array([x, y]))
    
    # Row 3 (top): 1 point
    points.append(0.5 * (v0 + v1 + v2))
    
    return np.array(points).flatten()


def _random_initial_config(rng):
    """Generate a random initial configuration inside the triangle."""
    pts = []
    for _ in range(N):
        u, v = rng.random(), rng.random()
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        pts.append([u + 0.5 * v, _H * v])
    return np.array(pts).flatten()


def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # 130s to be safe within 150s limit
    
    best_pts = None
    best_score = -np.inf
    
    # Try symmetric configuration first
    x0_sym = _symmetric_initial_config()
    x0_sym = _project_to_triangle(x0_sym.reshape(N, 2)).flatten()
    
    results = []
    
    # Optimize from symmetric start
    res = minimize(_objective, x0_sym, method='SLSQP',
                   options={'maxiter': 500, 'ftol': 1e-12})
    if res.success or res.fun < _objective(x0_sym):
        pts = _project_to_triangle(res.x.reshape(N, 2))
        score = _min_triangle_area(pts)
        results.append((score, pts))
    
    # Multi-start with random configurations
    rng = np.random.default_rng(42)
    n_random = 30
    
    for i in range(n_random):
        if time.time() > deadline:
            break
        
        x0 = _random_initial_config(rng)
        x0 = _project_to_triangle(x0.reshape(N, 2)).flatten()
        
        # Add some perturbation
        x0_perturbed = x0 + 0.05 * rng.standard_normal(N * 2)
        x0_perturbed = _project_to_triangle(x0_perturbed.reshape(N, 2)).flatten()
        
        res = minimize(_objective, x0_perturbed, method='SLSQP',
                       options={'maxiter': 300, 'ftol': 1e-12})
        
        if res.fun < _objective(x0_perturbed):
            pts = _project_to_triangle(res.x.reshape(N, 2))
            score = _min_triangle_area(pts)
            results.append((score, pts))
    
    # Also try some intermediate random starts
    for i in range(15):
        if time.time() > deadline:
            break
        
        x0 = _random_initial_config(rng)
        x0 = _project_to_triangle(x0.reshape(N, 2)).flatten()
        
        res = minimize(_objective, x0, method='SLSQP',
                       options={'maxiter': 200, 'ftol': 1e-10})
        
        pts = _project_to_triangle(res.x.reshape(N, 2))
        score = _min_triangle_area(pts)
        results.append((score, pts))
    
    # Find the best result
    if results:
        best_score, best_pts = max(results, key=lambda x: x[0])
    else:
        # Fallback: return symmetric configuration
        best_pts = _project_to_triangle(_symmetric_initial_config().reshape(N, 2))
    
    return best_pts
