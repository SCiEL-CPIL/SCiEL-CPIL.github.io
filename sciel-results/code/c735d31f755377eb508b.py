"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0  # safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _center_distance(vertsA, vertsB):
    """Approximate center distance between two polygons."""
    cxA = sum(v[0] for v in vertsA) / len(vertsA)
    cyA = sum(v[1] for v in vertsA) / len(vertsA)
    cxB = sum(v[0] for v in vertsB) / len(vertsB)
    cyB = sum(v[1] for v in vertsB) / len(vertsB)
    return math.hypot(cxA - cxB, cyA - cyB)

def _separating_axis_check(vertsA, vertsB, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    if _center_distance(vertsA, vertsB) > 4.0:
        return False
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    arrA = np.array(vertsA)
    arrB = np.array(vertsB)
    for axis in axes:
        ax, ay = axis
        projA = arrA[:, 0] * ax + arrA[:, 1] * ay
        projB = arrB[:, 0] * ax + arrB[:, 1] * ay
        if projA.max() <= projB.min() + tol or projB.max() <= projA.min() + tol:
            return False
    return True

def _min_overlap_depth(vertsA, vertsB):
    """Compute minimum overlap depth between two convex polygons (0 if disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    arrA = np.array(vertsA)
    arrB = np.array(vertsB)
    max_overlap = 0.0
    for axis in axes:
        ax, ay = axis
        projA = arrA[:, 0] * ax + arrA[:, 1] * ay
        projB = arrB[:, 0] * ax + arrB[:, 1] * ay
        overlap = min(projA.max(), projB.max()) - max(projA.min(), projB.min())
        if overlap > max_overlap:
            max_overlap = overlap
        if overlap < -1e-9:
            return 0.0
    return max_overlap

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    normals_arr = np.array(normals)
    cx, cy = center
    all_verts = []
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        all_verts.extend(verts)
    all_verts_arr = np.array(all_verts)
    rel = all_verts_arr - np.array([cx, cy])
    projs = rel @ normals_arr.T
    return np.max(projs) / apothem_unit

def _is_valid(centers_angles, tol=1e-9):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    arr_polys = [np.array(p) for p in polys]
    for i in range(m):
        for j in range(i + 1, m):
            # Quick center distance check
            cA = arr_polys[i].mean(axis=0)
            cB = arr_polys[j].mean(axis=0)
            if np.linalg.norm(cA - cB) > 4.0:
                continue
            if _separating_axis_check(polys[i], polys[j], tol):
                return False
    return True

def _smooth_collision_penalty(all_verts, beta=20.0):
    """Smooth collision penalty using soft-min of overlap depths."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _min_overlap_depth(all_verts[i], all_verts[j])
            if depth > 0:
                penalty += depth * depth
    return penalty

def _smooth_outer_side(cfg, center, angle_deg, beta=15.0):
    """Smooth outer side length using log-sum-exp over all vertex-projection pairs."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    normals_arr = np.array(normals)
    cx, cy = center
    all_verts = []
    for (px, py, pangle) in cfg:
        verts = _hex_vertices(px, py, pangle)
        all_verts.extend(verts)
    all_verts_arr = np.array(all_verts)
    rel = all_verts_arr - np.array([cx, cy])
    projs = rel @ normals_arr.T
    vals = projs.flatten() / apothem_unit
    max_val = np.max(vals)
    shifted = vals - max_val
    lse = max_val + (1.0 / beta) * np.log(np.sum(np.exp(beta * shifted)))
    return lse

def _golden_section_minimize(f, a, b, tol=1e-7, max_iter=200):
    """Golden-section search for minimum of f on [a, b]."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
    return (a + b) / 2.0

def _hex_normals(angle_deg):
    """Precomputed edge normals for a unit hexagon with given rotation."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    normals = []
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        rnx = ca * nx - sa * ny
        rny = sa * nx + ca * ny
        normals.append((rnx, rny))
    return normals

def _optimize_outer_rotation(cfg, center, time_budget=5.0):
    """
    Treat the 12 hexagons as a fixed composite body.
    Optimize the outer hexagon rotation angle via golden-section search.
    The outer hexagon center is also optimized (2D) for each rotation.
    Returns (best_center, best_angle, best_side).
    """
    t_start = time.time()
    
    # Precompute all vertices of all inner hexagons
    all_verts = []
    for (px, py, pangle) in cfg:
        verts = _hex_vertices(px, py, pangle)
        all_verts.extend(verts)
    
    # Compute centroid of composite body
    cx_body = sum(v[0] for v in all_verts) / len(all_verts)
    cy_body = sum(v[1] for v in all_verts) / len(all_verts)
    
    apothem_unit = math.sqrt(3) / 2.0
    
    def side_for_rotation_and_center(angle_deg, cx, cy):
        a = math.radians(angle_deg)
        normals = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        max_need = 0.0
        for (vx, vy) in all_verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
        return max_need
    
    def _chebyshev_center(normals, all_verts, init_cx, init_cy, max_iter=100, tol=1e-12):
        """
        Find the Chebyshev center (minimize max projection) using subgradient method.
        """
        normals_arr = np.array(normals)
        verts_arr = np.array(all_verts)
        cx, cy = init_cx, init_cy
        step = 1.0
        for it in range(max_iter):
            rel = verts_arr - np.array([cx, cy])
            projs = (rel @ normals_arr.T) / apothem_unit
            max_val = np.max(projs)
            # Active constraints: those within tolerance of max
            active_mask = projs > max_val - 1e-8
            if not np.any(active_mask):
                break
            # Subgradient: sum of active normals (direction to move center)
            active_normals = normals_arr[active_mask]
            # Weighted by how close to max
            weights = (projs[active_mask] - (max_val - 1e-8)) + 1e-8
            grad = (active_normals.T @ weights) / weights.sum()
            cx += step * grad[0]
            cy += step * grad[1]
            step *= 0.95
            if step < tol:
                break
        
        return cx, cy
    
    def side_for_rotation(angle_deg):
        """For a given outer rotation, find optimal center via Chebyshev center."""
        a = math.radians(angle_deg)
        normals = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        # Find Chebyshev center
        cx, cy = _chebyshev_center(normals, all_verts, cx_body, cy_body)
        
        return side_for_rotation_and_center(angle_deg, cx, cy), (cx, cy)
    
    # Golden section search over rotation [0, 60) degrees
    def f_rot(theta):
        s, _ = side_for_rotation(theta)
        return s
    
    best_angle = _golden_section_minimize(f_rot, 0.0, 60.0, tol=1e-7, max_iter=100)
    best_side, best_center = side_for_rotation(best_angle)
    
    # Multi-start refinement: try nearby angles
    for offset in [-0.5, 0.5, -1.0, 1.0, -2.0, 2.0]:
        test_angle = (best_angle + offset) % 60.0
        test_side, test_center = side_for_rotation(test_angle)
        if test_side < best_side:
            best_side = test_side
            best_center = test_center
            best_angle = test_angle
    
    return best_center, best_angle, best_side

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, tuple(center), angle)
    
    def make_hex_lattice():
        """3x4 hexagonal lattice with offset."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts
    
    def make_ring():
        """12 hexagons in a ring, oriented outward."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts
    
    def make_flower():
        """Flower pattern: 1 center + 6 inner ring + 5 outer."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2 * math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_dense_cluster():
        """Tight cluster: center + 6 around + 5 more."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = math.sqrt(3) * math.cos(theta)
            y = math.sqrt(3) * math.sin(theta)
            pts.append([x, y, 0.0])
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = 2 * math.sqrt(3) * math.cos(theta)
            y = 2 * math.sqrt(3) * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_hexagonal_12():
        """12 hexagons: center + 5 inner + 6 outer."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * i / 5 + math.pi/5
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2 * math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_tight_hex12():
        """Tight hexagonal arrangement: 1 center + 6 ring + 5 outer, all aligned."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2.0 * math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_compact_grid():
        """Compact 2-row arrangement with slight rotation for better packing."""
        pts = []
        # Row 1: 6 hexagons
        for i in range(6):
            x = i * 1.86  # slightly less than 2*sqrt(3)/2 to allow rotation
            y = 0.0
            pts.append([x - 2.79, y, 0.0])
        # Row 2: 6 hexagons, offset
        for i in range(6):
            x = i * 1.86 + 0.93
            y = math.sqrt(3) * 0.95  # slightly less than sqrt(3)
            pts.append([x - 2.79, y, 0.0])
        return pts
    
    def make_hcp_12():
        """Hexagonal close-packed: 4-4-4 in three staggered rows."""
        pts = []
        row_h = math.sqrt(3)
        col_w = 2.0
        for r in range(3):
            offset = col_w / 2 if r % 2 else 0.0
            for c in range(4):
                x = c * col_w + offset
                y = r * row_h
                pts.append([x - 3.0, y - row_h, 0.0])
        return pts
    
    def make_optimal_hex12():
        """Optimized 12-hexagon: 1 center + 6 ring + 5 outer with tuned radii."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3) * 0.999
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2.0 * math.sqrt(3) * 0.999
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def smooth_objective(v, beta=15.0):
        """
        Smooth objective: smooth outer side + collision penalty.
        v = [x1,y1,a1, ..., x12,y12,a12, cx,cy, outer_angle]
        """
        cfg = []
        for i in range(N):
            cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
        center = (v[3*N], v[3*N+1])
        outer_angle = v[3*N+2]
        
        all_verts = []
        for p in cfg:
            verts = _hex_vertices(p[0], p[1], p[2])
            all_verts.append(verts)
        
        smooth_side = _smooth_outer_side(cfg, center, outer_angle, beta=beta)
        collision_penalty = _smooth_collision_penalty(all_verts, beta=20.0)
        
        # Add a small penalty for large displacements to keep things bounded
        disp_penalty = 0.0
        for i in range(N):
            disp_penalty += v[3*i]**2 + v[3*i+1]**2
        disp_penalty += v[3*N]**2 + v[3*N+1]**2
        disp_penalty *= 0.01
        
        return smooth_side + 50.0 * collision_penalty + disp_penalty
    
    def gradient_descent_polish(cfg_init, center_init, angle_init, time_budget, beta=15.0):
        """L-BFGS-B polish with smooth objective."""
        def flatten(cfg, center, angle):
            v = []
            for p in cfg:
                v.extend([p[0], p[1], p[2]])
            v.extend([center[0], center[1], angle])
            return np.array(v, dtype=float)
        
        def unflatten(v):
            cfg = []
            for i in range(N):
                cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
            center = [v[3*N], v[3*N+1]]
            angle = v[3*N+2]
            return cfg, center, angle
        
        def obj(v):
            if time.time() - t0 > TIME_LIMIT - 2.0:
                return 1e6
            return smooth_objective(v, beta=beta)
        
        v = flatten(cfg_init, center_init, angle_init)
        
        result = minimize(
            obj, v, method='L-BFGS-B',
            options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        best_v = v.copy()
        best_f = smooth_objective(v, beta=beta)
        
        if result.fun < best_f:
            best_v = result.x.copy()
            best_f = result.fun
        
        return best_v, best_f
    
    # Try multiple initial configurations
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    inits = [make_hex_lattice(), make_ring(), make_flower(), make_dense_cluster(), 
             make_hexagonal_12(), make_tight_hex12(), make_compact_grid(),
             make_hcp_12(), make_optimal_hex12()]
    for init in inits:
        s = full_score(init, (0.0, 0.0), 0.0)
        if s is not None and s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0
    
    if best_cfg is None:
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0
    
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score
    
    # Simulated annealing phase
    T = 1.0
    T_end = 0.001
    alpha = 0.9995
    
    sa_time_limit = TIME_LIMIT - 40.0  # Reserve 40s for polishing + composite body
    
    # Precompute lattice positions for snap moves
    lattice_positions = []
    row_height = math.sqrt(3)
    col_width = 2.0
    for r in range(5):
        for c in range(6):
            x = c * col_width + (col_width / 2 if r % 2 else 0.0)
            y = r * row_height
            lattice_positions.append((x - 3.0, y - 2.0 * row_height))
    
    while time.time() - t0 < sa_time_limit:
        r = random.random()
        
        if r < 0.40:
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.15 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.55:
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 4.0 + 2.0 * T)
        elif r < 0.65:
            old_cx, old_cy = center[0], center[1]
            step = 0.08 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.75:
            old_angle = angle
            angle += random.gauss(0, 1.5 + 1.5 * T)
        elif r < 0.85:
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.12 * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
        elif r < 0.92:
            # Triple move
            idx1, idx2, idx3 = random.sample(range(N), 3)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            old_x3, old_y3 = cfg[idx3][0], cfg[idx3][1]
            step = 0.10 * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
            cfg[idx3][0] += random.gauss(0, step)
            cfg[idx3][1] += random.gauss(0, step)
        else:
            # Snap move: move one hexagon toward a nearby lattice position
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            best_lp = None
            best_dist = float('inf')
            for lx, ly in lattice_positions:
                d = (cfg[idx][0] - lx)**2 + (cfg[idx][1] - ly)**2
                if d < best_dist:
                    best_dist = d
                    best_lp = (lx, ly)
            if best_lp is not None:
                cfg[idx][0] = cfg[idx][0] + 0.8 * (best_lp[0] - cfg[idx][0])
                cfg[idx][1] = cfg[idx][1] + 0.8 * (best_lp[1] - cfg[idx][1])
        
        new_score = full_score(cfg, tuple(center), angle)
        
        if new_score is not None:
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                if r < 0.40:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.55:
                    cfg[idx][2] = old_ang
                elif r < 0.65:
                    center[0], center[1] = old_cx, old_cy
                elif r < 0.75:
                    angle = old_angle
                elif r < 0.85:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
                elif r < 0.92:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
                    cfg[idx3][0], cfg[idx3][1] = old_x3, old_y3
                else:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
        else:
            if r < 0.40:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.55:
                cfg[idx][2] = old_ang
            elif r < 0.65:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.75:
                angle = old_angle
            elif r < 0.85:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
            elif r < 0.92:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
                cfg[idx3][0], cfg[idx3][1] = old_x3, old_y3
            else:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
        
        T = max(T_end, T * alpha)
    
    # Composite body optimization: treat 12 hexagons as rigid, optimize outer rotation
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 12.0:
        try:
            cb_center, cb_angle, cb_side = _optimize_outer_rotation(
                best_cfg, best_center, time_budget=min(remaining_time * 0.3, 8.0)
            )
            if cb_side < best_score:
                # Verify validity
                if _is_valid(best_cfg):
                    best_score = cb_side
                    best_center = cb_center
                    best_angle = cb_angle
        except Exception:
            pass
    
    # Multi-stage gradient descent polish
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 15.0:
        try:
            # Stage 1: Coarse polish
            polished_v, polished_f = gradient_descent_polish(
                best_cfg, best_center, best_angle, min(remaining_time * 0.4, 15.0), beta=15.0
            )
            cfg_polished = []
            for i in range(N):
                cfg_polished.append([polished_v[3*i], polished_v[3*i+1], polished_v[3*i+2]])
            center_polished = [polished_v[3*N], polished_v[3*N+1]]
            angle_polished = polished_v[3*N+2]
            
            polished_score = full_score(cfg_polished, tuple(center_polished), angle_polished)
            if polished_score is not None and polished_score < best_score:
                best_score = polished_score
                best_cfg = cfg_polished
                best_center = center_polished
                best_angle = angle_polished
        except Exception:
            pass
    
    # Stage 2: Fine polish with tighter tolerance
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 10.0:
        try:
            polished_v2, polished_f2 = gradient_descent_polish(
                best_cfg, best_center, best_angle, min(remaining_time * 0.5, 12.0), beta=30.0
            )
            cfg_polished2 = []
            for i in range(N):
                cfg_polished2.append([polished_v2[3*i], polished_v2[3*i+1], polished_v2[3*i+2]])
            center_polished2 = [polished_v2[3*N], polished_v2[3*N+1]]
            angle_polished2 = polished_v2[3*N+2]
            
            polished_score2 = full_score(cfg_polished2, tuple(center_polished2), angle_polished2)
            if polished_score2 is not None and polished_score2 < best_score:
                best_score = polished_score2
                best_cfg = cfg_polished2
                best_center = center_polished2
                best_angle = angle_polished2
        except Exception:
            pass
    
    # Stage 3: Very fine polish
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 8.0:
        try:
            polished_v3, polished_f3 = gradient_descent_polish(
                best_cfg, best_center, best_angle, min(remaining_time * 0.5, 8.0), beta=60.0
            )
            cfg_polished3 = []
            for i in range(N):
                cfg_polished3.append([polished_v3[3*i], polished_v3[3*i+1], polished_v3[3*i+2]])
            center_polished3 = [polished_v3[3*N], polished_v3[3*N+1]]
            angle_polished3 = polished_v3[3*N+2]
            
            polished_score3 = full_score(cfg_polished3, tuple(center_polished3), angle_polished3)
            if polished_score3 is not None and polished_score3 < best_score:
                best_score = polished_score3
                best_cfg = cfg_polished3
                best_center = center_polished3
                best_angle = angle_polished3
        except Exception:
            pass
    
    # Final composite body optimization after all polishing
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 8.0:
        try:
            cb_center2, cb_angle2, cb_side2 = _optimize_outer_rotation(
                best_cfg, best_center, time_budget=min(remaining_time * 0.4, 5.0)
            )
            if cb_side2 < best_score:
                if _is_valid(best_cfg):
                    best_score = cb_side2
                    best_center = cb_center2
                    best_angle = cb_angle2
        except Exception:
            pass
    
    # Final quick refinement
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 3.0:
        for _ in range(500):
            if time.time() - t0 > TIME_LIMIT - 1.0:
                break
            cfg_try = [row[:] for row in best_cfg]
            center_try = best_center[:]
            angle_try = best_angle
            move_type = random.random()
            if move_type < 0.5:
                idx = random.randrange(N)
                cfg_try[idx][0] += random.gauss(0, 0.015)
                cfg_try[idx][1] += random.gauss(0, 0.015)
            elif move_type < 0.7:
                idx = random.randrange(N)
                cfg_try[idx][2] += random.gauss(0, 1.0)
            elif move_type < 0.85:
                center_try[0] += random.gauss(0, 0.01)
                center_try[1] += random.gauss(0, 0.01)
            else:
                angle_try += random.gauss(0, 0.5)
            score_try = full_score(cfg_try, tuple(center_try), angle_try)
            if score_try is not None and score_try < best_score:
                best_score = score_try
                best_cfg = cfg_try
                best_center = center_try
                best_angle = angle_try
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
