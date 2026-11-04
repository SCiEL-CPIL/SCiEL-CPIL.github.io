"""Optimized Heilbronn convex regions n=14."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math
from itertools import combinations

N = 14
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _triangle_area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Minimum triangle area over all triples."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _convex_hull_area(points):
    """Area of convex hull using Andrew's monotone chain."""
    pts = np.array(points)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    n = len(pts)
    if n < 3:
        return 0.0
    lower = []
    for p in pts:
        while len(lower) >= 2:
            x1, y1 = lower[-1] - lower[-2]
            x2, y2 = p - lower[-1]
            if x1[0] * y2[1] - x1[1] * y2[0] <= 1e-12:
                lower.pop()
            else:
                break
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2:
            x1, y1 = upper[-1] - upper[-2]
            x2, y2 = p - upper[-1]
            if x1[0] * y2[1] - x1[1] * y2[0] <= 1e-12:
                upper.pop()
            else:
                break
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    # Shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area

def _score(points):
    """Normalized score: min triangle area / convex hull area."""
    cha = _convex_hull_area(points)
    if cha < 1e-10:
        return 0.0
    mta = _min_triangle_area(points)
    return mta / cha

def _soft_min_triangle_area(points, beta=50.0):
    """Smooth approximation of min triangle area using log-sum-exp."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                areas.append(a)
    areas = np.array(areas)
    # soft min: -1/beta * log(sum(exp(-beta*a)))
    min_a = np.min(areas)
    shifted = -beta * (areas - min_a)
    # clip for numerical stability
    shifted = np.clip(shifted, -500, 500)
    log_sum = min_a - (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    return log_sum

def _neg_soft_min_normalized(x, beta=50.0):
    """Objective for optimization: negative of soft-min triangle area / hull area."""
    points = x.reshape(-1, 2)
    # Normalize to unit area to make objective scale-invariant
    cha = _convex_hull_area(points)
    if cha < 1e-10:
        return 0.0
    points_norm = points / math.sqrt(cha)
    sm = _soft_min_triangle_area(points_norm, beta)
    return -sm

def _neg_min_normalized(x, beta=100.0):
    """Objective with exact min for polishing."""
    points = x.reshape(-1, 2)
    cha = _convex_hull_area(points)
    if cha < 1e-10:
        return 0.0
    points_norm = points / math.sqrt(cha)
    mta = _min_triangle_area(points_norm)
    return -mta

def _regular_polygon(n, radius=1.0):
    """Generate n points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _lattice_init(n):
    """Generate points on a grid pattern."""
    # Try to fit n points in a roughly square grid
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    points = []
    for i in range(n):
        r = i // cols
        c = i % cols
        points.append([c / max(cols - 1, 1), r / max(rows - 1, 1)])
    return np.array(points[:n])

def _quasi_random_init(n, seed=42):
    """Low-discrepancy sequence using golden ratio."""
    phi = (1 + math.sqrt(5)) / 2
    points = []
    for i in range(n):
        x = (i / phi) % 1.0
        y = ((i * i) / phi) % 1.0
        points.append([x, y])
    return np.array(points)

def construct():
    start_time = time.time()
    
    # Multiple initializations
    inits = [
        _regular_polygon(N),
        _lattice_init(N),
        _quasi_random_init(N),
        np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
                  [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5],
                  [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],
                  [0.5, 0.5], [0.5, 0.05]]),
    ]
    
    best_score = -float('inf')
    best_points = inits[0]
    
    # Evaluate initial points
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()
    
    # Differential evolution for global search
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 60:
        try:
            bounds = [(0, 1), (0, 1)] * N
            # Use multiple starting points for DE
            de_result = differential_evolution(
                _neg_soft_min_normalized,
                bounds,
                args=(50.0,),
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='sobol',
                update_interval=1,
                callback=lambda xk, convergence: None
            )
            if de_result.fun < 0:
                de_points = de_result.x.reshape(-1, 2)
                s = _score(de_points)
                if s > best_score:
                    best_score = s
                    best_points = de_points.copy()
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        # Local polish with exact min
        try:
            res = minimize(
                _neg_min_normalized,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < 0:
                polish_points = res.x.reshape(-1, 2)
                s = _score(polish_points)
                if s > best_score:
                    best_score = s
                    best_points = polish_points.copy()
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        # Final high-precision polish
        try:
            res = minimize(
                _neg_min_normalized,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if res.fun < 0:
                polish_points = res.x.reshape(-1, 2)
                s = _score(polish_points)
                if s > best_score:
                    best_score = s
                    best_points = polish_points.copy()
        except Exception:
            pass
    
    # Ensure points are within [0,1]^2
    best_points = np.clip(best_points, 0, 1)
    
    return best_points
# EVOLVE-BLOCK-END
