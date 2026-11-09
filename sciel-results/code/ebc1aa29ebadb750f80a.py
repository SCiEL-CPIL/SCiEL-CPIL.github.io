import numpy as np
import time
from scipy.spatial import ConvexHull

N = 13

def _min_triangle_area_and_indices(points):
    """Return (min_area, (i,j,k)) for the minimum-area triangle."""
    n = len(points)
    min_area = np.inf
    min_idx = (0, 1, 2)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
    return min_area, min_idx

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def get_score(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2):
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area, _ = _min_triangle_area_and_indices(points)
    return min_area / hull_area

def _subgradient_step(points, lr, hull_area):
    """
    Take a subgradient ascent step on min triangle area / hull area.
    Since we want to maximize min_area / hull_area, and hull_area changes slowly,
    we focus on increasing the minimum triangle area.
    """
    min_area, (i, j, k) = _min_triangle_area_and_indices(points)
    if min_area < 1e-12:
        # Degenerate: push points apart randomly
        rng = np.random.default_rng()
        points = points + rng.normal(0, 0.01, points.shape)
        return points
    
    a = points[i]
    b = points[j]
    c = points[k]
    
    # Area = 0.5 * |cross(b-a, c-a)|
    # Let v = b - a, w = c - a
    # area = 0.5 * |v_x * w_y - v_y * w_x|
    # The sign tells us the orientation
    cross_val = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    sign = np.sign(cross_val) if cross_val != 0 else 1.0
    
    # Gradient of 0.5*|cross| w.r.t. each point
    # d/d a: -sign * 0.5 * (perp of (b-a) cross (c-a) contribution)
    # Actually: area = 0.5 * sign * cross_val (since we take abs, and sign is fixed locally)
    # d(area)/da = 0.5 * sign * (-d(cross_val)/da)
    # cross_val = (b0-a0)(c1-a1) - (b1-a1)(c0-a0)
    # d(cross_val)/da0 = -(c1-a1)
    # d(cross_val)/da1 = (c0-a0)
    # So d(area)/da = 0.5*sign * (c1-a1, -(c0-a0))
    # d(cross_val)/db0 = (c1-a1)
    # d(cross_val)/db1 = -(c0-a0)
    # d(area)/db = 0.5*sign * (c1-a1, -(c0-a0))... wait let me redo
    
    # cross_val = (b0-a0)(c1-a1) - (b1-a1)(c0-a0)
    # Let u = b-a, v = c-a
    # cross_val = u0*v1 - u1*v0
    # d(cross_val)/du0 = v1, d(cross_val)/du1 = -v0
    # d(cross_val)/dv0 = -u1, d(cross_val)/dv1 = u0
    # d(cross_val)/da = -d(cross_val)/du - d(cross_val)/dv
    #   = -(v1, -v0) - (-u1, u0) = (-v1+u1, v0-u0)
    # d(cross_val)/db = d(cross_val)/du = (v1, -v0)
    # d(cross_val)/dc = d(cross_val)/dv = (-u1, u0)
    
    u = b - a
    v = c - a
    
    grad_a = -np.array([v[1] - u[1], -(v[0] - u[0])]) * 0.5 * sign
    # Wait, let me be more careful:
    # d(cross)/da0 = -v1, d(cross)/da1 = v0  => d(cross)/da = (-v1, v0)
    # d(cross)/db0 = v1,  d(cross)/db1 = -v0 => d(cross)/db = (v1, -v0)
    # d(cross)/dc0 = -u1, d(cross)/dc1 = u0 => d(cross)/dc = (-u1, u0)
    
    grad_a = np.array([-v[1], v[0]]) * 0.5 * sign
    grad_b = np.array([v[1], -v[0]]) * 0.5 * sign
    grad_c = np.array([-u[1], u[0]]) * 0.5 * sign
    
    new_points = points.copy()
    new_points[i] += lr * grad_a
    new_points[j] += lr * grad_b
    new_points[k] += lr * grad_c
    
    return new_points

def _optimize_from(points, deadline, max_iter=500):
    """
    Subgradient ascent to maximize min triangle area.
    """
    points = points.copy()
    best_points = points.copy()
    best_score = get_score(points)
    
    lr = 0.01
    n_no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            break
        
        new_points = _subgradient_step(points, lr, hull_area)
        
        new_score = get_score(new_points)
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            n_no_improve = 0
            # Slightly increase lr on improvement
            lr = min(lr * 1.05, 0.05)
        else:
            n_no_improve += 1
            lr *= 0.95
            if n_no_improve > 50:
                break
        
        points = new_points
    
    return best_points, best_score

def _generate_regular_polygon_configs():
    """Generate structured initial configurations."""
    configs = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Regular 12-gon + center
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    configs.append(np.vstack([pts12, np.array([[0.0, 0.0]])]))
    
    # Regular 11-gon + 2 inner
    angles11 = np.linspace(0, 2*np.pi, 11, endpoint=False)
    pts11 = np.column_stack([np.cos(angles11), np.sin(angles11)])
    configs.append(np.vstack([pts11, np.array([[0.5, 0.0], [-0.5, 0.0]])]))
    
    # Regular 10-gon + 3 inner (triangle at center)
    angles10 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    pts10 = np.column_stack([np.cos(angles10), np.sin(angles10)])
    inner_angles = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    inner = np.column_stack([0.4*np.cos(inner_angles), 0.4*np.sin(inner_angles)])
    configs.append(np.vstack([pts10, inner]))
    
    # Regular 9-gon + 4 inner (square at center)
    angles9 = np.linspace(0, 2*np.pi, 9, endpoint=False)
    pts9 = np.column_stack([np.cos(angles9), np.sin(angles9)])
    inner_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    inner = np.column_stack([0.35*np.cos(inner_angles), 0.35*np.sin(inner_angles)])
    configs.append(np.vstack([pts9, inner]))
    
    # Two concentric rings: 7 outer, 6 inner
    angles7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(angles7), np.sin(angles7)])
    angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    inner = np.column_stack([0.5*np.cos(angles6), 0.5*np.sin(angles6)])
    configs.append(np.vstack([outer, inner]))
    
    # 8 outer, 5 inner
    angles8 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    outer = np.column_stack([np.cos(angles8), np.sin(angles8)])
    angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    inner = np.column_stack([0.45*np.cos(angles5), 0.45*np.sin(angles5)])
    configs.append(np.vstack([outer, inner]))
    
    # Random with seed
    rng = np.random.default_rng(42)
    for seed in [1, 7, 13, 99]:
        rng = np.random.default_rng(seed)
        pts = rng.random((N, 2))
        configs.append(pts)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    configs = _generate_regular_polygon_configs()
    
    best_score = -np.inf
    best_points = None
    
    for idx, init_points in enumerate(configs):
        if time.time() > deadline - 5.0:
            break
        
        try:
            optimized, score = _optimize_from(init_points, deadline - 5.0)
            if score > best_score:
                best_score = score
                best_points = optimized
        except Exception:
            pass
    
    if best_points is None:
        best_points = configs[0]
    
    return best_points
