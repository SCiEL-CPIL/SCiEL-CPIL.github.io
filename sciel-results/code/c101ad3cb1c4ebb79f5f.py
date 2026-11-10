# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

# Precompute all triangle index combinations (286 triangles for 13 points)
N_POINTS = 13
N_TRI = 286
_TRI_IDX = np.array(list(itertools.combinations(range(N_POINTS), 3)), dtype=np.int32)

# For fast local search: precompute which triangles involve each point
_POINT_TRIS = [[] for _ in range(N_POINTS)]
for i, tri in enumerate(_TRI_IDX):
    for p in tri:
        _POINT_TRIS[p].append(i)
_POINT_TRIS = [np.array(x) for x in _POINT_TRIS]

def _triangle_areas(points):
    """Compute areas of all 286 triangles (vectorized)."""
    p1 = points[_TRI_IDX[:, 0]]
    p2 = points[_TRI_IDX[:, 1]]
    p3 = points[_TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _update_areas_fast(points, areas, moved_idx):
    """Update areas for triangles involving moved_idx (fast path)."""
    tris = _POINT_TRIS[moved_idx]
    p1 = points[_TRI_IDX[tris, 0]]
    p2 = points[_TRI_IDX[tris, 1]]
    p3 = points[_TRI_IDX[tris, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas[tris] = 0.5 * np.abs(cross)
    return areas

def compute_score(points):
    """Compute the Heilbronn score: A_min / A_hull"""
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 0:
        return 0.0
    areas = _triangle_areas(points)
    return float(np.min(areas) / A_hull)

def _soft_min_score(points, beta=50.0):
    """Differentiable surrogate: soft-min of areas / hull area."""
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e10
    if A_hull <= 0:
        return 1e10
    areas = _triangle_areas(points)
    m = np.min(areas)
    soft_min = m - np.log(np.sum(np.exp(-beta * (areas - m)))) / beta
    return -soft_min / A_hull

def _fast_local_search(points, areas, deadline, max_iters=50000):
    """Fast local search: move one point at a time, recompute only affected triangles."""
    best_score = np.min(areas)
    pts = points.copy()
    a = areas.copy()
    
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for step in step_sizes:
        if time.time() > deadline - 5:
            break
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N_POINTS):
                if time.time() > deadline - 5:
                    break
                # Try 8 directions
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        old_x, old_y = pts[i]
                        pts[i] = [old_x + dx, old_y + dy]
                        _update_areas_fast(pts, a, i)
                        new_min = np.min(a)
                        if new_min > best_score + 1e-12:
                            best_score = new_min
                            improved = True
                        else:
                            pts[i] = [old_x, old_y]
                            _update_areas_fast(pts, a, i)
    
    return pts, a

def _nelder_mead_soft(points, start_time, deadline, max_iter=3000, beta=50.0):
    """Nelder-Mead optimization on the soft-min surrogate."""
    x0 = points.flatten()
    
    def obj(x):
        if time.time() > deadline - 5:
            return 1e10
        pts = x.reshape(N_POINTS, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        return _soft_min_score(pts, beta=beta)
    
    try:
        result = minimize(
            obj, x0, method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-7,
                'fatol': 1e-10,
                'adaptive': True,
            }
        )
        return result.x.reshape(N_POINTS, 2)
    except Exception:
        return points

def _nelder_mead_hard(points, start_time, deadline, max_iter=2000):
    """Nelder-Mead on the hard objective (actual min area / hull area)."""
    x0 = points.flatten()
    
    def obj(x):
        if time.time() > deadline - 5:
            return 1e10
        pts = x.reshape(N_POINTS, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        s = compute_score(pts)
        return -s if s > 0 else 1e10
    
    try:
        result = minimize(
            obj, x0, method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-8,
                'fatol': 1e-11,
                'adaptive': True,
            }
        )
        return result.x.reshape(N_POINTS, 2)
    except Exception:
        return points

def _final_polish(points, start_time, deadline):
    """Final high-precision polish: alternating soft-min NM with increasing beta + fast local search."""
    best_pts = points.copy()
    best_score = compute_score(points)
    
    betas = [30, 50, 100, 200, 400]
    for beta in betas:
        if time.time() > deadline - 10:
            break
        # Nelder-Mead with this beta
        refined = _nelder_mead_soft(best_pts, start_time, deadline, max_iter=1500, beta=beta)
        s = compute_score(refined)
        if s > best_score:
            best_score = s
            best_pts = refined.copy()
        
        # Fast local search
        areas = _triangle_areas(best_pts)
        best_pts, areas = _fast_local_search(best_pts, areas, deadline, max_iters=20000)
        s = compute_score(best_pts)
        if s > best_score:
            best_score = s
    
    # Final hard NM polish
    if time.time() > deadline - 30:
        return best_pts
    refined = _nelder_mead_hard(best_pts, start_time, deadline, max_iter=1000)
    s = compute_score(refined)
    if s > best_score:
        best_pts = refined.copy()
    
    return best_pts

def heilbronn_convex13() -> np.ndarray:
    """
    Construct 13 points maximizing A_min / A_hull.
    Uses multi-start with fast local search, soft-min Nelder-Mead, and final polish.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    def time_left():
        return deadline - time.time()
    
    # --- Initial configurations ---
    
    def make_config1():
        """Regular 12-gon + center"""
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        outer = np.column_stack([np.cos(angles), np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        return np.vstack([outer, center])
    
    def make_config2():
        """Two concentric rings (7 outer, 6 inner)"""
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
        outer = np.column_stack([np.cos(a1), np.sin(a1)])
        inner = np.column_stack([0.55 * np.cos(a2), 0.55 * np.sin(a2)])
        return np.vstack([outer, inner])
    
    def make_config3():
        """Three rings (5, 4, 4)"""
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        p1 = np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = np.column_stack([0.6 * np.cos(a2), 0.6 * np.sin(a2)])
        p3 = np.column_stack([0.3 * np.cos(a3), 0.3 * np.sin(a3)])
        return np.vstack([p1, p2, p3])
    
    def make_config4():
        """1 center, 6 at r=0.5, 6 at r=1.0"""
        a1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
        p1 = np.array([[0.0, 0.0]])
        p2 = np.column_stack([0.5 * np.cos(a1), 0.5 * np.sin(a1)])
        p3 = np.column_stack([np.cos(a2), np.sin(a2)])
        return np.vstack([p1, p2, p3])
    
    def make_config5():
        """Perturbed regular 13-gon"""
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        rng = np.random.default_rng(42)
        points += rng.normal(0, 0.03, size=points.shape)
        return points
    
    def make_config6():
        """1 center, 3 at r=0.3, 3 at r=0.6, 6 at r=1.0"""
        a1 = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 3
        a3 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        p1 = np.array([[0.0, 0.0]])
        p2 = np.column_stack([0.3 * np.cos(a1), 0.3 * np.sin(a1)])
        p3 = np.column_stack([0.6 * np.cos(a2), 0.6 * np.sin(a2)])
        p4 = np.column_stack([np.cos(a3), np.sin(a3)])
        return np.vstack([p1, p2, p3, p4])
    
    initial_configs = [
        make_config1(),
        make_config2(),
        make_config3(),
        make_config4(),
        make_config5(),
        make_config6(),
    ]
    
    # Add random starts
    rng = np.random.default_rng(7)
    for _ in range(6):
        pts = rng.uniform(-1, 1, size=(N_POINTS, 2))
        initial_configs.append(pts)
    
    best_points = None
    best_score = 0.0
    
    # --- Phase 1: Evaluate and refine each initial config ---
    for config in initial_configs:
        if time_left() < 60:
            break
        
        points = config.copy()
        score = compute_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Fast local search (very fast, many iterations)
        if time_left() > 60:
            areas = _triangle_areas(points)
            points, areas = _fast_local_search(points, areas, deadline, max_iters=30000)
            score = compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
        
        # Nelder-Mead with soft-min surrogate
        if time_left() > 90:
            points = _nelder_mead_soft(points, start_time, deadline, max_iter=1500, beta=50)
            score = compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
            
            # Another fast local search after NM
            if time_left() > 60:
                areas = _triangle_areas(points)
                points, areas = _fast_local_search(points, areas, deadline, max_iters=20000)
                score = compute_score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
    
    # --- Phase 2: Final polish on best ---
    if best_points is not None and time_left() > 120:
        best_points = _final_polish(best_points, start_time, deadline)
        best_score = compute_score(best_points)
    
    # --- Phase 3: More polish rounds if time remains ---
    if best_points is not None and time_left() > 60:
        for _ in range(3):
            if time_left() < 30:
                break
            best_points = _final_polish(best_points, start_time, deadline)
            s = compute_score(best_points)
            if s > best_score:
                best_score = s
    
    # Ensure validity
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = make_config1()
    
    try:
        hull = ConvexHull(best_points)
    except Exception:
        best_points = make_config1()
    
    return best_points

# EVOLVE-BLOCK-END
