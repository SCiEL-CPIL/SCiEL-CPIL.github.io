"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

def _all_triangle_areas(points):
    """Compute all triangle areas vectorized."""
    n = len(points)
    idx = np.array(list(combinations(range(n), 3)))
    pi = points[:, idx[:, 0]]
    pj = points[:, idx[:, 1]]
    pk = points[:, idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        if len(points) < 3:
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    areas = _all_triangle_areas(points)
    min_tri = np.min(areas)
    return min_tri / hull_area

def _soft_min(areas, beta):
    """
    Smooth approximation of min using log-sum-exp with negative sign.
    -1/beta * log(sum(exp(-beta * areas)))
    As beta -> inf, this approaches min(areas).
    """
    # Shift for numerical stability
    shifted = areas - np.min(areas)
    return np.min(areas) - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))

def _neg_soft_score(points, beta):
    """Negative soft score for minimization."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 1e6
    areas = _all_triangle_areas(points)
    sm = _soft_min(areas, beta)
    return -(sm / hull_area)

def _neg_exact_score(points):
    """Negative exact score for minimization."""
    return -_score(points)

def _regular_polygon(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _two_ring(n):
    """Two concentric rings with alternating angles."""
    n1 = n // 2
    n2 = n - n1
    r1 = 1.0
    r2 = 0.5
    
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
    
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    
    return np.vstack([pts1, pts2])

def _hexagonal_lattice(n):
    """Points arranged in a hexagonal-like pattern."""
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # First ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    # Second ring (12 points, but we only need up to n)
    for i in range(12):
        angle = i * np.pi / 6
        r = 2.0 if i % 2 == 0 else np.sqrt(3)
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    
    pts = np.array(pts[:n])
    # Normalize to unit area
    hull_area = _convex_hull_area(pts)
    if hull_area > 0:
        pts = pts * np.sqrt(1.0 / hull_area)
    return pts

def _random_initialization(rng, n):
    """Random points in unit square, then normalized."""
    pts = rng.uniform(-1, 1, size=(n, 2))
    hull_area = _convex_hull_area(pts)
    if hull_area > 0:
        pts = pts * np.sqrt(1.0 / hull_area)
    return pts

def _optimize_powell(points, deadline, beta=50.0, maxiter=1000):
    """
    Powell optimization with soft-min objective.
    Powell is derivative-free and handles non-smooth landscapes well.
    """
    def obj(x):
        pts = x.reshape(-1, 2)
        if time.time() > deadline - 1:
            return 1e6
        return _neg_soft_score(pts, beta)
    
    x0 = points.flatten()
    try:
        result = minimize(
            obj,
            x0,
            method='Powell',
            options={
                'maxiter': maxiter,
                'xtol': 1e-12,
                'ftol': 1e-14,
                'adaptive': True
            }
        )
        best_points = result.x.reshape(-1, 2)
        best_score = _score(best_points)
        return best_points, best_score
    except Exception:
        return points, _score(points)

def _optimize_nelder_mead(points, deadline, maxiter=5000):
    """Nelder-Mead local optimization with exact objective."""
    def obj(x):
        pts = x.reshape(-1, 2)
        if time.time() > deadline - 1:
            return 1e6
        return _neg_exact_score(pts)
    
    x0 = points.flatten()
    try:
        result = minimize(
            obj,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        best_points = result.x.reshape(-1, 2)
        best_score = _score(best_points)
        return best_points, best_score
    except Exception:
        return points, _score(points)

def _optimize_de(points, deadline, popsize=15, maxiter=50):
    """
    Differential Evolution for global search.
    Uses soft-min objective for smoother landscape.
    """
    def obj(x):
        pts = x.reshape(-1, 2)
        if time.time() > deadline - 1:
            return 1e6
        return _neg_soft_score(pts, beta=20.0)
    
    bounds = [(-2.0, 2.0)] * (2 * N)
    try:
        result = differential_evolution(
            obj,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-8,
            seed=42,
            polish=False,
            init='sobol',
            x0=points.flatten()
        )
        best_points = result.x.reshape(-1, 2)
        best_score = _score(best_points)
        return best_points, best_score
    except Exception:
        return points, _score(points)

def _fine_tune(points, deadline, max_iters=20000, initial_step=0.01):
    """
    Custom fine-tuning: random perturbations with adaptive step size.
    """
    points = points.copy()
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    rng = np.random.RandomState(123)
    step_size = initial_step
    no_improve = 0
    
    for _ in range(max_iters):
        if time.time() > deadline - 2:
            break
        
        # Perturb a random subset of points
        n_perturb = rng.randint(1, 4)
        idxs = rng.choice(N, size=n_perturb, replace=False)
        deltas = rng.normal(0, step_size, size=(n_perturb, 2))
        
        new_points = points.copy()
        new_points[idxs] += deltas
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            points = new_points
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            no_improve += 1
            if no_improve > 200:
                step_size *= 0.9
                no_improve = 0
                if step_size < 1e-6:
                    break
                    
    return best_points, best_score

def construct():
    start_time = time.time()
    # Safety margin: kill at 145s to be safe against 150s limit
    deadline = start_time + 145.0
    
    best_points = None
    best_score = -1.0
    
    # Generate multiple initializations
    inits = []
    inits.append(_regular_polygon(N))
    inits.append(_two_ring(N))
    inits.append(_hexagonal_lattice(N))
    
    rng = np.random.RandomState(42)
    for _ in range(3):
        inits.append(_random_initialization(rng, N))
    
    # Phase 1: Powell optimization from multiple initializations
    for init_pts in inits:
        if time.time() > deadline - 60:
            break
        
        pts, sc = _optimize_powell(init_pts, deadline, beta=50.0, maxiter=500)
        if sc > best_score:
            best_score = sc
            best_points = pts
    
    # If Powell didn't find anything good, try DE
    if best_points is None or best_score < 0.001:
        if time.time() < deadline - 30:
            pts, sc = _optimize_de(inits[0], deadline, popsize=15, maxiter=50)
            if sc > best_score:
                best_score = sc
                best_points = pts
    
    # Fallback
    if best_points is None or best_score < 0.0001:
        best_points = _regular_polygon(N)
        best_score = _score(best_points)
    
    # Phase 2: Local refinement with Nelder-Mead
    if time.time() < deadline - 30:
        pts, sc = _optimize_nelder_mead(best_points, deadline, maxiter=5000)
        if sc > best_score:
            best_points = pts
            best_score = sc
    
    # Phase 3: Fine tuning
    if time.time() < deadline - 10:
        pts, sc = _fine_tune(best_points, deadline, max_iters=15000, initial_step=0.05)
        if sc > best_score:
            best_points = pts
            best_score = sc
    
    # Phase 4: Final polish with Nelder-Mead
    if time.time() < deadline - 2:
        pts, sc = _optimize_nelder_mead(best_points, deadline, maxiter=2000)
        if sc > best_score:
            best_points = pts
            best_score = sc
    
    # Final check: ensure we return a valid array
    if best_points is None:
        best_points = _regular_polygon(N)
        
    return best_points
