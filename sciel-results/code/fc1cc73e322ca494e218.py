"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 115.0  # safety margin below 150s

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

def _separating_axis_check(vertsA, vertsB):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    # Quick rejection by center distance
    cxA = sum(v[0] for v in vertsA) / 6
    cyA = sum(v[1] for v in vertsA) / 6
    cxB = sum(v[0] for v in vertsB) / 6
    cyB = sum(v[1] for v in vertsB) / 6
    if (cxA - cxB)**2 + (cyA - cyB)**2 > 12.0:  # max touching distance is 2*sqrt(3) ~ 3.46, squared ~ 12
        return False
    
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j]):
                return False
    return True

def _sdf_hex_point(px, py, angle_deg, side=1.0):
    """Signed distance from point (px,py) to hexagon centered at origin with rotation angle_deg.
    Returns negative if inside, positive if outside."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    lx = ca * px + sa * py
    ly = -sa * px + ca * py
    apothem = side * math.sqrt(3) / 2.0
    
    # Check if inside using half-plane tests
    max_proj = 0.0
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx, ny = math.cos(theta), math.sin(theta)
        proj = nx * lx + ny * ly
        if proj > max_proj:
            max_proj = proj
    
    if max_proj <= apothem + 1e-12:
        # Inside or on boundary
        return -(apothem - max_proj)
    else:
        # Outside: find minimum distance to edges
        min_d = float('inf')
        for k in range(6):
            t1 = math.radians(60 * k)
            t2 = math.radians(60 * (k + 1))
            v1 = (side * math.cos(t1), side * math.sin(t1))
            v2 = (side * math.cos(t2), side * math.sin(t2))
            dx, dy = v2[0] - v1[0], v2[1] - v1[1]
            len2 = dx * dx + dy * dy
            if len2 < 1e-12:
                continue
            t = max(0.0, min(1.0, ((lx - v1[0]) * dx + (ly - v1[1]) * dy) / len2))
            qx, qy = v1[0] + t * dx, v1[1] + t * dy
            d = math.hypot(lx - qx, ly - qy)
            if d < min_d:
                min_d = d
        return min_d

def _min_distance_hexagons(cfg_i, cfg_j):
    """Compute minimum distance between two hexagons. Positive if disjoint, negative if overlapping."""
    verts_i = _hex_vertices(cfg_i[0], cfg_i[1], cfg_i[2])
    verts_j = _hex_vertices(cfg_j[0], cfg_j[1], cfg_j[2])
    
    min_d = float('inf')
    for (vx, vy) in verts_i:
        d = _sdf_hex_point(vx - cfg_j[0], vy - cfg_j[1], cfg_j[2])
        if d < min_d:
            min_d = d
    for (vx, vy) in verts_j:
        d = _sdf_hex_point(vx - cfg_i[0], vy - cfg_i[1], cfg_i[2])
        if d < min_d:
            min_d = d
    return min_d

def _smooth_collision_penalty(cfg, sigma=0.03):
    """Smooth collision penalty using soft-min over pairwise distances."""
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _min_distance_hexagons(cfg[i], cfg[j])
            if d < 3.0 * sigma:
                # Smooth penalty: exp(-d/sigma) - 1 when d < 0 (overlapping), small when d > 0
                total += math.exp(-d / sigma) - 1.0
    return total

def _smooth_objective(v, t0):
    """Smooth objective: outer side + collision penalty."""
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = (v[3*N], v[3*N+1])
    angle = v[3*N+2]
    
    s = _outer_hex_side(cfg, center, angle)
    penalty = _smooth_collision_penalty(cfg)
    
    return s + 200.0 * penalty

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def make_hex_lattice():
        """3x4 hexagonal lattice, compact."""
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
        """12 hexagons in a ring around center."""
        pts = []
        R = 2.8
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts
    
    def make_cluster():
        """Central hex + ring of 6 + outer 5."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3) + 0.01
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2.0 * math.sqrt(3) + 0.01
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_compact_grid():
        """2x6 compact arrangement with hexagonal offsets."""
        pts = []
        dx = math.sqrt(3) + 0.01
        dy = 1.5 + 0.01
        for r in range(2):
            for c in range(6):
                x = c * dx + (dx/2 if r % 2 else 0)
                y = r * dy
                pts.append([x - 5*dx/2, y - dy/2, 0.0])
        return pts
    
    def make_two_rings():
        """Inner ring of 6 + outer ring of 6."""
        pts = []
        R1 = math.sqrt(3) + 0.01
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2 * math.sqrt(3) + 0.01
        for i in range(6):
            theta = 2 * math.pi * (i + 0.5) / 6
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, center, angle)
    
    # Generate multiple initial configurations
    inits = [make_hex_lattice(), make_ring(), make_cluster(), make_compact_grid(), make_two_rings()]
    
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    for init in inits:
        for out_angle in [0, 15, 30, 45, 60]:
            s = full_score(init, (0.0, 0.0), float(out_angle))
            if s is not None and s < best_score:
                best_score = s
                best_cfg = [row[:] for row in init]
                best_center = [0.0, 0.0]
                best_angle = float(out_angle)
    
    if best_cfg is None:
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-4, 4), random.uniform(-4, 4), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0
    
    # Phase 1: Simulated Annealing
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score
    
    T = 0.3
    T_end = 0.001
    alpha_sa = 0.9997
    
    sa_time_budget = 45.0
    while time.time() - t0 < sa_time_budget:
        r = random.random()
        
        if r < 0.50:
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.08 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.70:
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 4.0 + 3.0 * T)
        elif r < 0.80:
            old_cx, old_cy = center[0], center[1]
            step = 0.04 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.90:
            old_angle = angle
            angle += random.gauss(0, 1.5 + 1.5 * T)
        else:
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.06 * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
        
        new_score = full_score(cfg, tuple(center), angle)
        
        accepted = False
        if new_score is not None:
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                accepted = True
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
        
        if not accepted:
            if r < 0.50:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.70:
                cfg[idx][2] = old_ang
            elif r < 0.80:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.90:
                angle = old_angle
            else:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        
        T = max(T_end, T * alpha_sa)
    
    # Phase 2: L-BFGS-B polish with smooth objective
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 20.0:
        try:
            v0 = []
            for p in best_cfg:
                v0.extend([p[0], p[1], p[2]])
            v0.extend([best_center[0], best_center[1], best_angle])
            
            result = minimize(
                _smooth_objective,
                v0,
                args=(t0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 8000}
            )
            
            if time.time() - t0 < TIME_LIMIT - 10.0:
                v_opt = result.x
                cfg_opt = []
                for i in range(N):
                    cfg_opt.append([v_opt[3*i], v_opt[3*i+1], v_opt[3*i+2]])
                center_opt = [v_opt[3*N], v_opt[3*N+1]]
                angle_opt = v_opt[3*N+2]
                
                opt_score = full_score(cfg_opt, tuple(center_opt), angle_opt)
                if opt_score is not None and opt_score < best_score:
                    best_score = opt_score
                    best_cfg = cfg_opt
                    best_center = center_opt
                    best_angle = angle_opt
        except Exception:
            pass
    
    # Phase 3: Second SA restart from best found, with smaller steps
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 20.0:
        cfg = [row[:] for row in best_cfg]
        center = best_center[:]
        angle = best_angle
        cur_score = best_score
        
        T = 0.1
        T_end = 0.0005
        alpha_sa = 0.9996
        sa_time_budget = 15.0
        sa_start = time.time()
        
        while time.time() - sa_start < sa_time_budget and time.time() - t0 < TIME_LIMIT - 10.0:
            r = random.random()
            
            if r < 0.60:
                idx = random.randrange(N)
                old_x, old_y = cfg[idx][0], cfg[idx][1]
                step = 0.03 * (1.0 + 0.3 * T)
                cfg[idx][0] += random.gauss(0, step)
                cfg[idx][1] += random.gauss(0, step)
            elif r < 0.80:
                idx = random.randrange(N)
                old_ang = cfg[idx][2]
                cfg[idx][2] += random.gauss(0, 1.5 + 1.0 * T)
            elif r < 0.90:
                old_cx, old_cy = center[0], center[1]
                step = 0.02 * (1.0 + 0.3 * T)
                center[0] += random.gauss(0, step)
                center[1] += random.gauss(0, step)
            else:
                old_angle = angle
                angle += random.gauss(0, 0.5 + 0.5 * T)
            
            new_score = full_score(cfg, tuple(center), angle)
            
            accepted = False
            if new_score is not None:
                if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                    accepted = True
                    cur_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_cfg = [row[:] for row in cfg]
                        best_center = center[:]
                        best_angle = angle
            
            if not accepted:
                if r < 0.60:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.80:
                    cfg[idx][2] = old_ang
                elif r < 0.90:
                    center[0], center[1] = old_cx, old_cy
                else:
                    angle = old_angle
            
            T = max(T_end, T * alpha_sa)
    
    # Phase 4: Final fine-tune of outer hexagon parameters only
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 3.0:
        try:
            best_local_score = best_score
            for out_angle in np.linspace(0, 60, 720):
                s = full_score(best_cfg, tuple(best_center), float(out_angle))
                if s is not None and s < best_local_score:
                    best_local_score = s
                    best_angle = float(out_angle)
            
            # Fine tune center
            for dx in np.linspace(-0.05, 0.05, 41):
                for dy in np.linspace(-0.05, 0.05, 41):
                    s = full_score(best_cfg, (best_center[0]+dx, best_center[1]+dy), best_angle)
                    if s is not None and s < best_local_score:
                        best_local_score = s
                        best_center = [best_center[0]+dx, best_center[1]+dy]
            
            if best_local_score < best_score:
                best_score = best_local_score
        except Exception:
            pass
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
