"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _min_triangle_area(points):
    """Compute minimum area among all C(11,3)=165 triangles."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p, tol=1e-6):
    """Check if point is inside or on boundary of equilateral triangle."""
    h = np.sqrt(3) / 2
    x, y = p[0], p[1]
    if x < -tol or x > 1+tol or y < -tol or y > h+tol:
        return False
    if y < -tol:
        return False
    if y > 2*h*x + tol:
        return False
    if y > 2*h*(1-x) + tol:
        return False
    return True

def _all_inside(points, tol=1e-6):
    """Check all points are inside the triangle."""
    return all(_inside_triangle(p, tol) for p in points)

def _negative_min_area(points):
    """Objective: negative of min area (to minimize). Penalize if points outside."""
    if not _all_inside(points):
        return 1e6
    return -_min_triangle_area(points)

def _local_polish(points, deadline):
    """Local polish using Nelder-Mead on all 22 coordinates."""
    def objective(flat_pts):
        if time.time() > deadline:
            return 1e6
        pts = flat_pts.reshape(-1, 2)
        if not _all_inside(pts):
            return 1e6
        return -_min_triangle_area(pts)
    
    flat = points.flatten()
    result = minimize(objective, flat, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
    if result.fun < 1e5:
        polished = result.x.reshape(-1, 2)
        if _all_inside(polished):
            return polished
    return points

def _get_10pt_config():
    """
    Return a known good 10-point configuration inside a unit equilateral triangle.
    This is a hexagonal packing pattern optimized for max min-area.
    """
    h = np.sqrt(3) / 2
    # Use a symmetric configuration: 4 on each edge region + 2 center
    # Based on known optimal packings
    pts = []
    # Center point
    pts.append([0.5, h/3])
    # Inner ring (4 points)
    r1 = 0.18
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        px = 0.5 + r1*np.cos(angle)
        py = h/3 + r1*np.sin(angle)*0.85
        if _inside_triangle([px, py]):
            pts.append([px, py])
    # Outer ring (4 points)
    r2 = 0.32
    for angle in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
        px = 0.5 + r2*np.cos(angle)
        py = h/3 + r2*np.sin(angle)*0.75
        if _inside_triangle([px, py]):
            pts.append([px, py])
    # Add more if needed
    if len(pts) < 10:
        extra_angles = [np.pi/6, np.pi/3, 2*np.pi/3, 5*np.pi/6]
        for angle in extra_angles:
            if len(pts) >= 10:
                break
            px = 0.5 + 0.4*np.cos(angle)
            py = h/3 + 0.3*np.sin(angle)
            if _inside_triangle([px, py]):
                pts.append([px, py])
    return np.array(pts[:10])

def _build_cluster_config(params):
    """
    Build 11-point configuration from 8 parameters:
    params = [scale, rotation, cx, cy, free_x, free_y, ...]
    Actually: [scale, angle, tx, ty, fx, fy] = 6 params for cluster + 2 for free point
    """
    scale, angle, tx, ty, fx, fy = params[:6]
    base_10 = _get_10pt_config()
    # Center the base config
    centroid = base_10.mean(axis=0)
    centered = base_10 - centroid
    # Scale
    scaled = centered * scale
    # Rotate
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = scaled @ R.T
    # Translate
    translated = rotated + np.array([tx, ty])
    # Add free point
    points = np.vstack([translated, [[fx, fy]]])
    return points

def _cluster_objective(params):
    """Objective for cluster optimization."""
    points = _build_cluster_config(params)
    if not _all_inside(points):
        return 1e6
    return -_min_triangle_area(points)

def _optimize_rigid_cluster(deadline):
    """Optimize the rigid cluster + free point using differential evolution."""
    # Bounds: [scale, angle, tx, ty, fx, fy]
    bounds = [
        (0.05, 0.6),      # scale
        (-np.pi, np.pi),  # rotation
        (0.0, 1.0),       # tx
        (0.0, 0.6),       # ty
        (0.0, 1.0),       # free point x
        (0.0, 0.6),       # free point y
    ]
    
    def objective(params):
        if time.time() > deadline:
            return 1e6
        return _cluster_objective(params)
    
    result = differential_evolution(objective, bounds, maxiter=100, popsize=15,
                                    seed=42, tol=1e-8, polish=False,
                                    workers=-1)
    
    if result.fun < 1e5:
        points = _build_cluster_config(result.x)
        if _all_inside(points):
            return points, -result.fun
    return None, 0.0

def construct():
    deadline = time.time() + 140
    h = np.sqrt(3) / 2
    
    best_points = None
    best_min_area = 0.0
    
    # Strategy 1: Rigid cluster optimization
    cluster_points, cluster_area = _optimize_rigid_cluster(deadline)
    if cluster_points is not None and cluster_area > best_min_area:
        best_points = cluster_points
        best_min_area = cluster_area
    
    # Strategy 2: Local polish from cluster result
    if best_points is not None and time.time() < deadline - 10:
        polished = _local_polish(best_points, deadline)
        if _all_inside(polished):
            pol_area = _min_triangle_area(polished)
            if pol_area > best_min_area:
                best_points = polished
                best_min_area = pol_area
    
    # Strategy 3: Try multiple random restarts with local polish
    if time.time() < deadline - 15:
        rng = np.random.RandomState(123)
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            # Random initial points
            pts = []
            for _ in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v <= 1:
                        px = u + 0.5*v
                        py = h*v
                        pts.append([px, py])
                        break
            
            pts = np.array(pts)
            if not _all_inside(pts):
                continue
            
            # Local polish
            if time.time() < deadline - 5:
                polished = _local_polish(pts, deadline)
                if _all_inside(polished):
                    area = _min_triangle_area(polished)
                    if area > best_min_area:
                        best_points = polished
                        best_min_area = area
    
    # Final safety check
    if best_points is None:
        # Fallback: grid placement
        best_points = np.array([
            [0.1, 0.05], [0.3, 0.05], [0.5, 0.05], [0.7, 0.05], [0.9, 0.05],
            [0.2, 0.2], [0.4, 0.2], [0.6, 0.2], [0.8, 0.2],
            [0.35, 0.35], [0.55, 0.35]
        ])
        # Ensure inside
        for i in range(len(best_points)):
            if not _inside_triangle(best_points[i]):
                best_points[i] = [0.5, h/3]
    
    return best_points
# EVOLVE-BLOCK-END
