"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# Deadline guard
DEADLINE = 115.0  # seconds, leave 5s safety margin before 120s kill

def _min_triangle_area_fast(pts):
    """Compute the minimum triangle area among all triples of points."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            for k in range(j + 1, n):
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _soft_min_triangle_area(pts, temperature=0.02):
    """
    Smooth surrogate using log-sum-exp for the min triangle area.
    Returns a soft approximation to min(areas) that is differentiable.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            for k in range(j + 1, n):
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    areas = np.array(areas)
    T = temperature
    # soft_min = -T * log(sum(exp(-areas/T)))
    # For numerical stability:
    max_neg = np.max(-areas / T)
    log_sum = max_neg + np.log(np.sum(np.exp(-areas / T - max_neg)))
    soft_min = -T * log_sum
    return soft_min

def _project_to_triangle(px, py):
    """Project a point onto the interior of the equilateral triangle."""
    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2
    # Clip to triangle using half-plane constraints
    if py < 0:
        py = 0.0
    if py > sqrt3 * px:
        px = py / sqrt3
    if py > sqrt3 * (1 - px):
        px = 1 - py / sqrt3
    # Re-check
    if py < 0:
        py = 0.0
    if px < 0:
        px = 0.0
        py = 0.0
    if px > 1:
        px = 1.0
        py = 0.0
    if py < 0:
        py = 0.0
    if py > h:
        py = h
        px = 0.5
    if sqrt3 * px - py < 0:
        px = max(py / sqrt3, 0) + 1e-10
    if sqrt3 * (1 - px) - py < 0:
        px = min(1 - py / sqrt3, 1) - 1e-10
    return px, py

def construct():
    start_time = time.time()
    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2
    
    best_points = None
    best_score = -float('inf')
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Phase 1: Differential Evolution with soft-min surrogate
    temperature = 0.02
    
    def de_objective(x):
        pts = x.reshape(N, 2)
        # Soft penalty for points outside triangle
        penalty = 0.0
        for i in range(N):
            px, py = pts[i]
            if py < 0:
                penalty += abs(py) * 100
            if sqrt3 * px - py < 0:
                penalty += abs(sqrt3 * px - py) * 100
            if sqrt3 * (1 - px) - py < 0:
                penalty += abs(sqrt3 * (1 - px) - py) * 100
        if penalty > 0.01:
            return 1e5 + penalty
        # Use soft-min for smoother landscape
        soft_min = _soft_min_triangle_area(pts, temperature)
        return -soft_min
    
    # Define bounds for each coordinate
    bounds = [(0, 1)] * N + [(0, h)] * N
    
    # Run differential evolution
    try:
        result = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=150,
            popsize=12,
            tol=1e-7,
            mutation=(0.3, 1.0),
            recombination=0.9,
            polish=False,
            init='latinhypercube'
        )
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            pts = result.x.reshape(N, 2)
            proj_pts = np.array([_project_to_triangle(p[0], p[1]) for p in pts])
            score = _min_triangle_area_fast(proj_pts)
            update_best(proj_pts, score)
            return best_points
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 25:
        if best_points is not None:
            return best_points
        rng = np.random.RandomState(42)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                px = u + 0.5 * v
                py = 0.5 * sqrt3 * v
                pts.append([px, py])
        return np.array(pts)
    
    # Get the best from DE
    de_pts = result.x.reshape(N, 2)
    proj_de_pts = np.array([_project_to_triangle(p[0], p[1]) for p in de_pts])
    de_score = _min_triangle_area_fast(proj_de_pts)
    update_best(proj_de_pts, de_score)
    
    # Phase 2: Nelder-Mead polish on the exact min triangle area
    for restart in range(5):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        if restart == 0:
            x0 = proj_de_pts.flatten()
        else:
            rng = np.random.RandomState(42 + restart)
            perturbation = rng.randn(N, 2) * 0.05
            x0 = (proj_de_pts + perturbation).flatten()
        
        def nm_objective(x):
            pts = x.reshape(N, 2)
            penalty = 0.0
            for i in range(N):
                px, py = pts[i]
                if py < 0:
                    penalty += abs(py) * 1000
                if sqrt3 * px - py < 0:
                    penalty += abs(sqrt3 * px - py) * 1000
                if sqrt3 * (1 - px) - py <  0:
                    penalty += abs(sqrt3 * (1 - px) - py) * 1000
            if penalty > 0.001:
                return 1e4 + penalty
            min_area = _min_triangle_area_fast(pts)
            return -min_area
        
        try:
            nm_result = minimize(
                nm_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            nm_pts = nm_result.x.reshape(N, 2)
            proj_nm_pts = np.array([_project_to_triangle(p[0], p[1]) for p in nm_pts])
            nm_score = _min_triangle_area_fast(proj_nm_pts)
            update_best(proj_nm_pts, nm_score)
        except Exception:
            continue
    
    if best_points is not None:
        return best_points
    
    # Fallback: random points
    rng = np.random.RandomState(42)
    pts = []
    while len(pts) < N:
        u, v = rng.random(), rng.random()
        if u + v < 1:
            px = u + 0.5 * v
            py = 0.5 * sqrt3 * v
            pts.append([px, py])
    return np.array(pts)
# EVOLVE-BLOCK-END
