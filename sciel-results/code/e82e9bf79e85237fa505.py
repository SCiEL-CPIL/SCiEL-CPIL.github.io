import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 14
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 115s safety margin before 120s kill

def _hull_area(pts):
    """Compute convex hull area of 2D points."""
    if len(pts) < 3:
        return 0.0
    try:
        h = ConvexHull(pts)
        return h.volume
    except Exception:
        # Fallback: brute force convex hull via gift wrapping or just return 1.0
        # Use a simple approach: if points are nearly collinear, area ~ 0
        # For safety, return a small positive number
        return 1e-10

def _min_triangle_area(pts):
    """Compute minimum area of any triangle from triples of points."""
    n = len(pts)
    min_area = float('inf')
    # Vectorized approach for speed
    for i in range(n):
        for j in range(i+1, n):
            v1 = pts[j] - pts[i]
            # Cross product with all other points
            diffs = pts - pts[i]
            # Cross product: v1_x * dy - v1_y * dx
            cross = v1[0] * diffs[:, 1] - v1[1] * diffs[:, 0]
            # Exclude i and j
            areas = np.abs(cross) / 2.0
            areas[i] = float('inf')
            areas[j] = float('inf')
            m = np.min(areas)
            if m < min_area:
                min_area = m
    return min_area

def _score(pts):
    """Score = min_triangle_area / hull_area. Higher is better."""
    ha = _hull_area(pts)
    if ha < 1e-15:
        return 0.0
    mta = _min_triangle_area(pts)
    return mta / ha

def _objective(pts_flat):
    """Objective to minimize: negative of score."""
    pts = pts_flat.reshape(N, 2)
    s = _score(pts)
    if s <= 0:
        return 1e10
    return -s

def _make_initial_configs():
    """Generate several structured initial configurations."""
    configs = []
    
    # Config 1: Regular 14-gon on unit circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = 1.0
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    configs.append(pts)
    
    # Config 2: Two concentric regular 7-gons
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = angles_outer + np.pi/7
    outer = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    inner = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    configs.append(np.vstack([outer, inner]))
    
    # Config 3: Grid-like: 2 rows of 7
    x = np.linspace(0, 1, 7)
    y1 = np.full(7, 0.2)
    y2 = np.full(7, 0.8)
    pts = np.column_stack([np.concatenate([x, x]), np.concatenate([y1, y2])])
    configs.append(pts)
    
    # Config 4: Hexagonal packing pattern
    # 3 rows: 5, 5, 4 or similar
    pts = []
    # Row 1 (y=0): 5 points
    for i in range(5):
        pts.append([i * 0.25, 0.0])
    # Row 2 (y=0.2): 5 points, offset
    for i in range(5):
        pts.append([i * 0.25 + 0.125, 0.2])
    # Row 3 (y=0.4): 4 points
    for i in range(4):
        pts.append([i * 0.25 + 0.125, 0.4])
    configs.append(np.array(pts))
    
    # Config 5: Points on a square with interior
    # 4 corners, 4 midpoints of edges, 4 interior, 2 more
    pts = []
    # Corners
    pts.extend([[0,0],[1,0],[1,1],[0,1]])
    # Edge midpoints
    pts.extend([[0.5,0],[1,0.5],[0.5,1],[0,0.5]])
    # Interior points
    pts.extend([[0.25,0.25],[0.75,0.25],[0.25,0.75],[0.75,0.75]])
    # 2 more interior
    pts.extend([[0.5,0.5],[0.5,0.35]])
    configs.append(np.array(pts))
    
    # Config 6: Random but seeded
    rng = np.random.RandomState(42)
    configs.append(rng.rand(N, 2))
    
    # Config 7: Regular 14-gon scaled
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = 0.8
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    configs.append(pts)
    
    # Config 8: 3 concentric polygons: 6+5+3
    a1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    a3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    p1 = np.column_stack([1.0*np.cos(a1), 1.0*np.sin(a1)])
    p2 = np.column_stack([0.6*np.cos(a2), 0.6*np.sin(a2)])
    p3 = np.column_stack([0.25*np.cos(a3), 0.25*np.sin(a3)])
    configs.append(np.vstack([p1, p2, p3]))
    
    return configs

def construct():
    global _START_TIME, _DEADLINE
    best_pts = None
    best_score = -1.0
    
    # Fallback: random
    rng = np.random.RandomState(N)
    fallback = rng.rand(N, 2)
    
    configs = _make_initial_configs()
    
    for idx, pts in enumerate(configs):
        if time.time() > _DEADLINE:
            break
        
        # Evaluate initial score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        
        # Try local optimization
        try:
            res = minimize(_objective, pts.flatten(), method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < -best_score:
                opt_pts = res.x.reshape(N, 2)
                opt_score = _score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts.copy()
        except Exception:
            pass
        
        # Also try Nelder-Mead for robustness
        if time.time() > _DEADLINE:
            break
        try:
            res2 = minimize(_objective, pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res2.fun < -best_score:
                opt_pts2 = res2.x.reshape(N, 2)
                opt_score2 = _score(opt_pts2)
                if opt_score2 > best_score:
                    best_score = opt_score2
                    best_pts = opt_pts2.copy()
        except Exception:
            pass
    
    # If no valid result, return fallback
    if best_pts is None:
        return fallback
    
    return best_pts
