import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

_N = 13
# Precompute triangle indices
_I_IDX, _J_IDX, _K_IDX = [], [], []
for _i in range(_N):
    for _j in range(_i + 1, _N):
        for _k in range(_j + 1, _N):
            _I_IDX.append(_i)
            _J_IDX.append(_j)
            _K_IDX.append(_k)
_I_IDX = np.array(_I_IDX)
_J_IDX = np.array(_J_IDX)
_K_IDX = np.array(_K_IDX)

def _all_triangle_areas(pts):
    """Compute all 286 triangle areas vectorized."""
    x1, y1 = pts[_I_IDX, 0], pts[_I_IDX, 1]
    x2, y2 = pts[_J_IDX, 0], pts[_J_IDX, 1]
    x3, y3 = pts[_K_IDX, 0], pts[_K_IDX, 1]
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    return areas

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _exact_score(pts):
    """Compute A_min / A_hull exactly."""
    if len(pts) < 3:
        return 0.0
    areas = _all_triangle_areas(pts)
    a_min = np.min(areas)
    a_hull = _hull_area(pts)
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull

def _soft_min(areas, beta):
    """Smooth approximation of the minimum area."""
    max_area = np.max(areas)
    shifted = areas - max_area
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    return max_area - (1.0 / beta) * log_sum_exp

def _normalize_hull_area(pts):
    """Normalize points so the convex hull area is 1."""
    a_hull = _hull_area(pts)
    if a_hull <= 0 or not np.isfinite(a_hull):
        return pts
    # Area scales with square of linear dimensions
    scale = 1.0 / np.sqrt(a_hull)
    return pts * scale

def _delaunay_guided_relax(pts, n_iters=1000, step_size=0.01, time_limit=580.0, start_time=0.0):
    """
    Relax points using Delaunay triangulation to target small triangles.
    For each triangle in the Delaunay triangulation, if its area is small,
    push its vertices apart.
    """
    pts = pts.copy()
    best_score = _exact_score(pts)
    
    for it in range(n_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(pts)
        except Exception:
            break
        
        # Compute areas of Delaunay triangles
        tri_pts = pts[tri.simplices]
        x1, y1 = tri_pts[:, 0, 0], tri_pts[:, 0, 1]
        x2, y2 = tri_pts[:, 1, 0], tri_pts[:, 1, 1]
        x3, y3 = tri_pts[:, 2, 0], tri_pts[:, 2, 1]
        areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
        
        # Find the smallest triangle
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # Get the three vertices of the smallest triangle
        v0, v1, v2 = tri.simplices[min_idx]
        
        # Compute the centroid
        centroid = (pts[v0] + pts[v1] + pts[v2]) / 3.0
        
        # Push vertices outward from the centroid
        # The amount of push is proportional to how small the area is
        # relative to the average area
        avg_area = np.mean(areas)
        if avg_area > 0:
            push_factor = step_size * (1.0 - min_area / avg_area)
            push_factor = max(push_factor, 0.0)
        
        # Move each vertex away from the centroid
        for v in [v0, v1, v2]:
            direction = pts[v] - centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                pts[v] += push_factor * direction / norm
        
        # Also apply a small repulsive force between all pairs to prevent collapse
        for i in range(_N):
            for j in range(i + 1, _N):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 0.1:
                    force = 0.001 * (0.1 - dist) / dist
                    pts[i] += force * diff
                    pts[j] -= force * diff
        
        # Compute current score
        current_score = _exact_score(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
    
    if 'best_pts' in locals():
        return best_pts
    return pts

def _polish(pts, time_limit=580.0, start_time=0.0):
    """Polish the configuration using Nelder-Mead on the exact objective."""
    if time.time() - start_time > time_limit - 10:
        return pts
    
    def obj(x):
        p = x.reshape(_N, 2)
        s = _exact_score(p)
        if s <= 0 or not np.isfinite(s):
            return 1e10
        return -s
    
    try:
        res = minimize(
            obj,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if np.all(np.isfinite(res.x)):
            polished_pts = res.x.reshape(_N, 2)
            if _exact_score(polished_pts) > _exact_score(pts):
                return polished_pts
    except Exception:
        pass
    return pts

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Hexagonal lattice fragment
    # 13 points in a hexagonal arrangement
    pts = []
    # Center
    pts.append([0, 0])
    # First ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([np.cos(angle), np.sin(angle)])
    # Second ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        pts.append([2 * np.cos(angle), 2 * np.sin(angle)])
    configs.append(np.array(pts))
    
    # 3. Square grid with corners cut
    pts = []
    for i in range(4):
        for j in range(4):
            if (i, j) != (0, 0) and (i, j) != (3, 0) and (i, j) != (0, 3) and (i, j) != (3, 3):
                pts.append([i, j])
    # That's 12 points, add one more
    pts.append([1.5, 1.5])
    configs.append(np.array(pts))
    
    # 4. Random points in a circle
    rng = np.random.default_rng(42)
    angles = rng.uniform(0, 2 * np.pi, 13)
    radii = rng.uniform(0.3, 1.0, 13)
    configs.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # 5. Two concentric rings
    pts = []
    # Inner ring (5 points)
    for i in range(5):
        angle = i * 2 * np.pi / 5
        pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    # Outer ring (8 points)
    for i in range(8):
        angle = i * 2 * np.pi / 8 + np.pi / 8
        pts.append([np.cos(angle), np.sin(angle)])
    configs.append(np.array(pts))
    
    # 6. Triangular lattice
    pts = []
    for i in range(4):
        for j in range(4):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            if len(pts) < 13:
                pts.append([x, y])
    configs.append(np.array(pts[:13]))
    
    return configs

def _optimize_lattice(params, time_limit=580.0, start_time=0.0):
    """
    Optimize a hexagonal lattice model.
    params: [a, b, angle, cx, cy] where a, b are basis vector lengths,
    angle is the angle between them, cx, cy is the center offset.
    """
    a, b, angle, cx, cy = params
    
    # Generate 13 points in a hexagonal lattice
    pts = []
    # Define lattice vectors
    e1 = np.array([a, 0])
    e2 = np.array([b * np.cos(angle), b * np.sin(angle)])
    
    # Generate points in a hexagonal pattern
    # Center
    pts.append([cx, cy])
    
    # First shell: 6 points
    for i in range(6):
        # Use combinations of e1 and e2
        if i == 0:
            p = e1
        elif i == 1:
            p = e2
        elif i == 2:
            p = e1 + e2
        elif i == 3:
            p = -e1
        elif i == 4:
            p = -e2
        elif i == 5:
            p = -(e1 + e2)
        pts.append([cx + p[0], cy + p[1]])
    
    # Second shell: 6 more points
    for i in range(6):
        if i == 0:
            p = 2 * e1
        elif i == 1:
            p = 2 * e2
        elif i == 2:
            p = 2 * (e1 + e2)
        elif i == 3:
            p = -2 * e1
        elif i == 4:
            p = -2 * e2
        elif i == 5:
            p = -2 * (e1 + e2)
        pts.append([cx + p[0], cy + p[1]])
    
    pts = np.array(pts)
    
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    areas = _all_triangle_areas(pts)
    a_hull = _hull_area(pts)
    
    if a_hull <= 0 or not np.isfinite(a_hull):
        return 1e10
    
    beta = 50.0
    a_min_soft = _soft_min(areas, beta)
    score = a_min_soft / a_hull
    
    if not np.isfinite(score):
        return 1e10
    
    return -score

def heilbronn_convex13() -> np.ndarray:
    """
    Find 13 points maximizing A_min / A_hull.
    Uses Delaunay-guided relaxation and lattice optimization.
    """
    start_time = time.time()
    time_limit = 580  # Safety margin
    
    best_pts = None
    best_score = 0.0
    
    # Generate initial configurations
    configs = _generate_initial_configs()
    
    # Optimize each configuration using Delaunay-guided relaxation
    for i, config in enumerate(configs):
        if time.time() - start_time > time_limit - 60:
            break
        
        # Normalize to unit hull area
        config = _normalize_hull_area(config)
        
        # Apply Delaunay-guided relaxation
        relaxed = _delaunay_guided_relax(config, n_iters=500, step_size=0.01, 
                                          time_limit=time_limit, start_time=start_time)
        
        # Polish
        polished = _polish(relaxed, time_limit=time_limit, start_time=start_time)
        
        score = _exact_score(polished)
        if score > best_score:
            best_score = score
            best_pts = polished.copy()
    
    # Try lattice optimization
    if time.time() - start_time < time_limit - 60:
        # Initial guess for lattice parameters
        init_params = np.array([1.0, 1.0, np.pi / 3, 0.0, 0.0])
        
        try:
            res = minimize(
                _optimize_lattice,
                init_params,
                args=(time_limit, start_time),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if np.all(np.isfinite(res.x)):
                # Reconstruct points from optimal lattice parameters
                a, b, angle, cx, cy = res.x
                e1 = np.array([a, 0])
                e2 = np.array([b * np.cos(angle), b * np.sin(angle)])
                
                pts = []
                pts.append([cx, cy])
                for i in range(6):
                    if i == 0:
                        p = e1
                    elif i == 1:
                        p = e2
                    elif i == 2:
                        p = e1 + e2
                    elif i == 3:
                        p = -e1
                    elif i == 4:
                        p = -e2
                    elif i == 5:
                        p = -(e1 + e2)
                    pts.append([cx + p[0], cy + p[1]])
                
                for i in range(6):
                    if i == 0:
                        p = 2 * e1
                    elif i == 1:
                        p = 2 * e2
                    elif i == 2:
                        p = 2 * (e1 + e2)
                    elif i == 3:
                        p = -2 * e1
                    elif i == 4:
                        p = -2 * e2
                    elif i == 5:
                        p = -2 * (e1 + e2)
                    pts.append([cx + p[0], cy + p[1]])
                
                pts = np.array(pts)
                pts = _normalize_hull_area(pts)
                
                # Polish the lattice result
                polished = _polish(pts, time_limit=time_limit, start_time=start_time)
                score = _exact_score(polished)
                if score > best_score:
                    best_score = score
                    best_pts = polished.copy()
        except Exception:
            pass
    
    # Final polish if we have a good candidate
    if best_pts is not None and time.time() - start_time < time_limit - 10:
        final = _polish(best_pts, time_limit=time_limit, start_time=start_time)
        final_score = _exact_score(final)
        if final_score > best_score:
            best_pts = final
            best_score = final_score
    
    # Fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts
