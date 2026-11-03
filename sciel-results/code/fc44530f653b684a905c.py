"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize

N = 11

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _in_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    max_y = (3 ** 0.5) / 2 * min(x, 1 - x)
    if y > max_y + eps:
        return False
    return True

def _project_to_triangle(p):
    """Project a point onto the equilateral triangle."""
    x, y = p
    if y < 0:
        y = 0
    if x < 0:
        x = 0
    if x > 1:
        x = 1
    max_y = (3 ** 0.5) / 2 * min(x, 1 - x)
    if y > max_y:
        y = max_y
    return np.array([x, y])

def _initial_lattice():
    """Generate initial points using triangular lattice structure."""
    h = (3 ** 0.5) / 2
    points = []
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, h])
    points.append([0.5, 0.0])
    points.append([0.25, h/2])
    points.append([0.75, h/2])
    cx, cy = 0.5, h/3
    for i in range(5):
        angle = 2 * np.pi * i / 5 - np.pi/2
        r = 0.12
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        points.append([x, y])
    return np.array(points)

def _initial_random():
    """Generate random initial points using barycentric coordinates."""
    pts = []
    for _ in range(N):
        u = np.random.random()
        v = np.random.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        b1 = 1 - u - v
        b2 = u
        b3 = v
        x = b2 * 1 + b3 * 0.5
        y = b3 * (3 ** 0.5) / 2
        pts.append([x, y])
    return np.array(pts)

def _objective(params):
    """Objective: negative of min triangle area with penalty for out-of-bounds."""
    points = params.reshape(N, 2)
    penalty = 0.0
    for i in range(N):
        x, y = points[i]
        if y < 0:
            penalty += y * y
        if x < 0:
            penalty += x * x
        if x > 1:
            penalty += (x - 1) ** 2
        max_y = (3 ** 0.5) / 2 * min(max(x, 0), 1 - max(x, 0))
        if y > max_y:
            penalty += (y - max_y) ** 2
    if penalty > 1e-10:
        return penalty * 1000 + 100
    min_area = _min_triangle_area(points)
    return -min_area

def _run_local_optimization(initial_points, max_iter=2000):
    """Run local optimization using SLSQP with constraints."""
    params0 = initial_points.flatten()
    def obj(params):
        return _objective(params)
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (3**0.5)/2 * p[2*i] - p[2*i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (3**0.5)/2 * (1 - p[2*i]) - p[2*i+1]})
    bounds = [(0, 1)] * (2 * N)
    best_params = params0.copy()
    best_val = _min_triangle_area(initial_points)
    try:
        result = minimize(obj, params0, method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False})
        if result.fun < obj(params0):
            points_res = result.x.reshape(N, 2)
            valid = all(_in_triangle(points_res[i], eps=1e-6) for i in range(N))
            if valid:
                val = _min_triangle_area(points_res)
                if val > best_val:
                    best_val = val
                    best_params = result.x.copy()
    except Exception:
        pass
    return best_params.reshape(N, 2), best_val

def _nelder_mead_optimize(points, max_iter=5000):
    """Use Nelder-Mead for local optimization."""
    def obj(params):
        return _objective(params)
    params0 = points.flatten()
    try:
        result = minimize(obj, params0, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'disp': False})
        points_res = result.x.reshape(N, 2)
        for i in range(N):
            points_res[i] = _project_to_triangle(points_res[i])
        val = _min_triangle_area(points_res)
        return points_res, val
    except Exception:
        return points, _min_triangle_area(points)

def construct():
    """Construct 11 points in the equilateral triangle to maximize min triangle area."""
    np.random.seed(42)
    best_points = None
    best_val = -float('inf')
    configs = [_initial_lattice()]
    for _ in range(3):
        configs.append(_initial_random())
    for config in configs:
        pts, val = _run_local_optimization(config, max_iter=1500)
        if val > best_val:
            best_val = val
            best_points = pts
        pts2, val2 = _nelder_mead_optimize(pts, max_iter=3000)
        if val2 > best_val:
            best_val = val2
            best_points = pts2
    if best_points is not None:
        for i in range(N):
            best_points[i] = _project_to_triangle(best_points[i])
    return best_points
