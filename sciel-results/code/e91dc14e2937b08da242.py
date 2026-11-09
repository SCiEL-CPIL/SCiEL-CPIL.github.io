# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

_DEADLINE = time.time() + 600.0  # safety margin before 630s kill

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    """Compute A_min / A_hull."""
    ha = _hull_area(points)
    if ha <= 0:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0:
        return 0.0
    return ma / ha

def _objective(points_flat, n):
    """Objective: negative of min triangle area (we maximize min area, so minimize negative)."""
    points = points_flat.reshape(n, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    return -_min_triangle_area(points)

def _objective_ratio(points_flat, n):
    """Objective: negative of (min_area / hull_area)."""
    points = points_flat.reshape(n, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    s = _score(points)
    return -s

def _optimize_from_start(points0, n, max_iter=200, method='Nelder-Mead'):
    """Local optimization starting from points0."""
    x0 = points0.flatten()
    bounds = [(0.0, 1.0)] * (2 * n)
    
    best_x = x0.copy()
    best_val = _objective(x0, n)
    
    try:
        if method == 'Nelder-Mead':
            res = minimize(_objective, best_x, args=(n,), method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
        elif method == 'Powell':
            res = minimize(_objective, best_x, args=(n,), method='Powell',
                          bounds=bounds, options={'maxiter': max_iter, 'xtol': 1e-7, 'ftol': 1e-9})
        elif method == 'COBYLA':
            res = minimize(_objective, best_x, args=(n,), method='COBYLA',
                          bounds=bounds, options={'maxiter': max_iter, 'rhobeg': 0.01})
        else:
            res = minimize(_objective, best_x, args=(n,), method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    return best_x.reshape(n, 2)

def _optimize_ratio_from_start(points0, n, max_iter=200, method='Nelder-Mead'):
    """Local optimization maximizing the ratio directly."""
    x0 = points0.flatten()
    bounds = [(0.0, 1.0)] * (2 * n)
    
    best_x = x0.copy()
    best_val = _objective_ratio(x0, n)
    
    try:
        if method == 'Nelder-Mead':
            res = minimize(_objective_ratio, best_x, args=(n,), method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
        elif method == 'Powell':
            res = minimize(_objective_ratio, best_x, args=(n,), method='Powell',
                          bounds=bounds, options={'maxiter': max_iter, 'xtol': 1e-7, 'ftol': 1e-9})
        elif method == 'COBYLA':
            res = minimize(_objective_ratio, best_x, args=(n,), method='COBYLA',
                          bounds=bounds, options={'maxiter': max_iter, 'rhobeg': 0.01})
        else:
            res = minimize(_objective_ratio, best_x, args=(n,), method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    return best_x.reshape(n, 2)

def _regular_polygon(n, radius=0.45, center=(0.5, 0.5)):
    """Generate points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                           center[1] + radius * np.sin(angles)])

def _ring_with_interior(n, rng):
    """Generate points: outer ring + interior points."""
    # 9 on outer ring, 4 interior
    k_outer = 9
    k_inner = n - k_outer
    angles_outer = np.linspace(0, 2 * np.pi, k_outer, endpoint=False)
    pts = np.column_stack([0.5 + 0.45 * np.cos(angles_outer),
                           0.5 + 0.45 * np.sin(angles_outer)])
    
    # Interior points in a smaller ring or grid
    if k_inner == 1:
        pts = np.vstack([pts, [0.5, 0.5]])
    elif k_inner == 2:
        pts = np.vstack([pts, [0.5 + 0.15, 0.5], [0.5 - 0.15, 0.5]])
    elif k_inner == 3:
        angles_inner = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        inner = np.column_stack([0.5 + 0.18 * np.cos(angles_inner),
                                0.5 + 0.18 * np.sin(angles_inner)])
        pts = np.vstack([pts, inner])
    elif k_inner == 4:
        angles_inner = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        inner = np.column_stack([0.5 + 0.18 * np.cos(angles_inner),
                                0.5 + 0.18 * np.sin(angles_inner)])
        pts = np.vstack([pts, inner])
    else:
        # Fallback: random interior
        for _ in range(k_inner):
            r = rng.uniform(0, 0.25)
            a = rng.uniform(0, 2 * np.pi)
            pts = np.vstack([pts, [0.5 + r * np.cos(a), 0.5 + r * np.sin(a)]])
    
    return pts[:n]

def _two_rings(n, rng):
    """Generate points: two concentric rings."""
    # 7 on outer, 6 on inner
    k_outer = 7
    k_inner = n - k_outer
    angles_outer = np.linspace(0, 2 * np.pi, k_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, k_inner, endpoint=False) + np.pi / k_inner
    
    outer = np.column_stack([0.5 + 0.45 * np.cos(angles_outer),
                            0.5 + 0.45 * np.sin(angles_outer)])
    inner = np.column_stack([0.5 + 0.22 * np.cos(angles_inner),
                            0.5 + 0.22 * np.sin(angles_inner)])
    
    return np.vstack([outer, inner])[:n]

def _grid_jittered(n, rng):
    """Generate points on a jittered grid."""
    # Try 4x4 grid with one removed
    rows, cols = 4, 4
    pts = []
    for i in range(rows):
        for j in range(cols):
            if len(pts) >= n:
                break
            x = (j + 0.5) / cols + rng.uniform(-0.05, 0.05)
            y = (i + 0.5) / rows + rng.uniform(-0.05, 0.05)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pts.append([x, y])
    return np.array(pts[:n])

def _hexagonal_lattice(n, rng):
    """Generate points in a hexagonal lattice pattern."""
    pts = []
    # Hexagonal lattice points
    dx = 0.22
    dy = 0.22 * np.sqrt(3) / 2
    for row in range(5):
        for col in range(5):
            if len(pts) >= n:
                break
            x = 0.5 + (col - 2) * dx + (row % 2) * dx / 2
            y = 0.5 + (row - 2) * dy
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                pts.append([x, y])
    
    if len(pts) < n:
        # Fill with random points
        while len(pts) < n:
            pts.append([rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)])
    
    return np.array(pts[:n])

def _fejes_toth_approx(n, rng):
    """Approximate Fejes-Toth arrangement: regular polygon with optimized interior."""
    # Start with regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([0.5 + 0.48 * np.cos(angles),
                           0.5 + 0.48 * np.sin(angles)])
    return pts

def _best_known_13():
    """Return a known good configuration for 13 points."""
    # Based on known Heilbronn problem solutions
    # 13 points: try a configuration with good spread
    pts = []
    # 7 on outer heptagon
    for i in range(7):
        a = 2 * np.pi * i / 7
        pts.append([0.5 + 0.45 * np.cos(a), 0.5 + 0.45 * np.sin(a)])
    # 4 on inner square
    for i in range(4):
        a = 2 * np.pi * i / 4 + np.pi / 4
        pts.append([0.5 + 0.22 * np.cos(a), 0.5 + 0.22 * np.sin(a)])
    # 1 at center
    pts.append([0.5, 0.5])
    # 1 more interior
    pts.append([0.5 + 0.12, 0.5 + 0.12])
    
    return np.array(pts[:13])

def heilbronn_convex13() -> np.ndarray:
    n = 13
    rng = np.random.default_rng(seed=42)
    
    best_points = None
    best_score = -np.inf
    
    # Generate diverse initial configurations
    generators = [
        lambda: _regular_polygon(n, 0.45),
        lambda: _ring_with_interior(n, rng),
        lambda: _two_rings(n, rng),
        lambda: _grid_jittered(n, rng),
        lambda: _hexagonal_lattice(n, rng),
        lambda: _fejes_toth_approx(n, rng),
        lambda: _best_known_13(),
        lambda: rng.uniform(0.1, 0.9, (n, 2)),
        lambda: rng.uniform(0.1, 0.9, (n, 2)),
        lambda: rng.uniform(0.1, 0.9, (n, 2)),
    ]
    
    # Phase 1: Generate and score initial candidates
    candidates = []
    for gen in generators:
        if time.time() > _DEADLINE - 120:
            break
        try:
            pts = gen()
            if pts is None or len(pts) != n:
                continue
            if not np.all(np.isfinite(pts)):
                continue
            s = _score(pts)
            if s > 0:
                candidates.append((s, pts.copy()))
        except Exception:
            continue
    
    # Add more random candidates
    for _ in range(30):
        if time.time() > _DEADLINE - 120:
            break
        pts = rng.uniform(0.05, 0.95, (n, 2))
        s = _score(pts)
        if s > 0:
            candidates.append((s, pts.copy()))
    
    # Sort by score
    candidates.sort(key=lambda x: -x[0])
    
    # Phase 2: Local optimization on top candidates
    top_k = min(15, len(candidates))
    for i in range(top_k):
        if time.time() > _DEADLINE - 30:
            break
        s0, pts0 = candidates[i]
        
        # Try multiple optimization methods and perturbations
        for method in ['Nelder-Mead', 'Powell', 'COBYLA']:
            if time.time() > _DEADLINE - 20:
                break
            for attempt in range(2):
                if time.time() > _DEADLINE - 15:
                    break
                # Perturb for diversity
                if attempt == 0:
                    p_start = pts0.copy()
                else:
                    p_start = pts0 + rng.normal(0, 0.03, (n, 2))
                    p_start = np.clip(p_start, 0.02, 0.98)
                
                try:
                    # Optimize min area
                    p_opt1 = _optimize_from_start(p_start, n, max_iter=150, method=method)
                    if np.all(np.isfinite(p_opt1)):
                        s1 = _score(p_opt1)
                        if s1 > best_score:
                            best_score = s1
                            best_points = p_opt1.copy()
                    
                    # Optimize ratio
                    p_opt2 = _optimize_ratio_from_start(p_start, n, max_iter=150, method=method)
                    if np.all(np.isfinite(p_opt2)):
                        s2 = _score(p_opt2)
                        if s2 > best_score:
                            best_score = s2
                            best_points = p_opt2.copy()
                except Exception:
                    continue
        
        # Keep track of best raw candidate too
        if s0 > best_score:
            best_score = s0
            best_points = pts0.copy()
    
    # Final check: if we have a best candidate from raw generation that's better
    if candidates and (best_points is None or candidates[0][0] > best_score):
        best_score = candidates[0][0]
        best_points = candidates[0][1].copy()
    
    if best_points is None:
        # Fallback
        best_points = _regular_polygon(n, 0.45)
    
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(n, 0.45)
    
    return best_points

# EVOLVE-BLOCK-END
