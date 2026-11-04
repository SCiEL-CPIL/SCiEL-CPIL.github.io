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

def _separating_axis_check(vertsA, vertsB):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
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

def _pairwise_sat_distance(polyA, polyB):
    """Compute the minimum separation distance between two convex polygons using SAT.
    Returns a positive value if separated, negative if overlapping."""
    axes = _axes_from_polygon(polyA) + _axes_from_polygon(polyB)
    min_sep = float('inf')
    for axis in axes:
        minA, maxA = _project_polygon(polyA, axis)
        minB, maxB = _project_polygon(polyB, axis)
        sep = max(minB - maxA, minA - maxB)
        if sep < min_sep:
            min_sep = sep
    return min_sep

def _smooth_collision_penalty(centers_angles, margin=0.0):
    """
    Smooth penalty for hexagon overlaps using SAT distances.
    For each pair, compute the minimum separation distance (negative = overlap).
    Returns a smooth sum of squared overlaps.
    """
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    penalty = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            sep = _pairwise_sat_distance(polys[i], polys[j]) - margin
            if sep < 0:
                penalty += sep * sep
    return penalty

def _smooth_objective(v, margin=0.0, lam=50.0):
    """
    Smooth objective: outer_side + lambda * collision_penalty.
    v = [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11, cx,cy, outer_angle]
    """
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = (v[3*N], v[3*N+1])
    angle = v[3*N+2]
    
    side = _outer_hex_side(cfg, center, angle)
    penalty = _smooth_collision_penalty(cfg, margin)
    return side + lam * penalty

def _sdf_point_hex(px, py, cx, cy, angle_deg):
    """Signed distance from point (px,py) to hexagon centered at (cx,cy) with angle_deg.
    Negative inside, positive outside."""
    # Transform point to hexagon's local frame
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx, dy = px - cx, py - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    
    # Hexagon with flat top: vertices at angles 0, 60, 120, 180, 240, 300
    # Apothem = sqrt(3)/2
    apothem = math.sqrt(3) / 2.0
    
    # For a regular hexagon, the SDF can be computed by checking distance to each edge
    # and taking the minimum, with sign based on which side
    min_dist = float('inf')
    for k in range(6):
        theta1 = math.radians(60 * k)
        theta2 = math.radians(60 * (k + 1))
        # Edge from vertex k to vertex k+1
        x1, y1 = math.cos(theta1), math.sin(theta1)
        x2, y2 = math.cos(theta2), math.sin(theta2)
        
        # Edge vector
        ex, ey = x2 - x1, y2 - y1
        elen = math.hypot(ex, ey)
        ex, ey = ex / elen, ey / elen
        
        # Normal (inward for regular hexagon centered at origin)
        nx, ny = -ey, ex
        # Check if normal points inward
        if nx * x1 + ny * y1 > 0:
            nx, ny = -nx, -ny
        
        # Distance from point to edge line
        dist = nx * (lx - x1) + ny * (ly - y1)
        
        # Check if projection falls within edge
        t = (lx - x1) * ex + (ly - y1) * ey
        if t < 0:
            dist = math.hypot(lx - x1, ly - y1)
        elif t > elen:
            dist = math.hypot(lx - x2, ly - y2)
        
        # Sign: negative if inside (dist < 0 means on inward side)
        # Actually, for a convex polygon, we want the minimum over all edges
        # of the signed distance where negative = inside
        min_dist = min(min_dist, dist)
    
    return min_dist

def _sdf_hex_hex(p1, p2):
    """Approximate SDF between two hexagons using vertex-edge distances.
    Returns negative if overlapping, positive if separated."""
    # Use the SAT-based distance as a proxy
    poly1 = _hex_vertices(p1[0], p1[1], p1[2])
    poly2 = _hex_vertices(p2[0], p2[1], p2[2])
    return _pairwise_sat_distance(poly1, poly2)

def _nelder_mead_polish(cfg, center, angle, time_budget):
    """Use Nelder-Mead for final polishing."""
    def flatten(c, cen, ang):
        v = []
        for p in c:
            v.extend([p[0], p[1], p[2]])
        v.extend([cen[0], cen[1], ang])
        return np.array(v, dtype=float)
    
    def unflatten(v):
        v = list(v)
        c = []
        for i in range(N):
            c.append([v[3*i], v[3*i+1], v[3*i+2]])
        cen = [v[3*N], v[3*N+1]]
        ang = v[3*N+2]
        return c, cen, ang
    
    def objective(v):
        c, cen, ang = unflatten(v)
        # Check validity with SAT
        if not _is_valid(c):
            return 1e6
        return _outer_hex_side(c, tuple(cen), ang)
    
    v0 = flatten(cfg, center, angle)
    
    # Nelder-Mead with adaptive step size
    result = minimize(
        objective,
        v0,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
    )
    
    cfg_opt, cen_opt, ang_opt = unflatten(result.x)
    score = _outer_hex_side(cfg_opt, tuple(cen_opt), ang_opt)
    if _is_valid(cfg_opt):
        return cfg_opt, cen_opt, ang_opt, score
    return cfg, center, angle, _outer_hex_side(cfg, tuple(center), angle)

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
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
    
    def make_ring_initial():
        """12 hexagons arranged in a ring around the center."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts
    
    def make_strip_initial():
        """Two strips of 6 hexagons."""
        pts = []
        for i in range(6):
            pts.append([i * 1.9 - 5.0, -1.0, 0.0])
            pts.append([i * 1.9 - 5.0 + 0.95, 1.0, 30.0])
        return pts
    
    def make_rotated_grid():
        """Rotated hex grid by 30 degrees."""
        pts = make_hex_lattice()
        ca, sa = math.cos(math.radians(30)), math.sin(math.radians(30))
        new_pts = []
        for x, y, a in pts:
            new_pts.append([ca*x - sa*y, sa*x + ca*y, a + 30.0])
        return new_pts
    
    def make_compact_cluster():
        """Compact cluster: 2 rows of 6, tightly packed."""
        pts = []
        # Row 1: 6 hexagons
        for i in range(6):
            x = i * 1.85
            y = 0.0
            pts.append([x - 5.0, y, 0.0])
        # Row 2: 6 hexagons, offset
        for i in range(6):
            x = i * 1.85 + 0.925
            y = math.sqrt(3) * 0.95
            pts.append([x - 5.0, y, 0.0])
        return pts
    
    def make_hexagonal_arrangement():
        """12 hexagons in a hexagonal pattern (1+6+5 or similar)."""
        pts = []
        # Center hexagon
        pts.append([0, 0, 0])
        # Ring of 6
        R = 1.85
        for i in range(6):
            theta = math.radians(60 * i)
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            pts.append([x, y, 0])
        # Outer 5
        R2 = 3.6
        for i in range(5):
            theta = math.radians(60 * i + 30)
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0])
        return pts
    
    def flatten(cfg, center, angle):
        v = []
        for p in cfg:
            v.extend([p[0], p[1], p[2]])
        v.extend([center[0], center[1], angle])
        return np.array(v, dtype=float)
    
    def unflatten(v):
        v = list(v)
        cfg = []
        for i in range(N):
            cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
        center = [v[3*N], v[3*N+1]]
        angle = v[3*N+2]
        return cfg, center, angle
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, center, angle)
    
    # Generate diverse initial configurations
    inits = [
        make_hex_lattice(),
        make_ring_initial(),
        make_strip_initial(),
        make_rotated_grid(),
        make_compact_cluster(),
        make_hexagonal_arrangement()
    ]
    
    # Add randomized perturbations
    for _ in range(4):
        base = make_hex_lattice()
        perturbed = []
        for p in base:
            perturbed.append([
                p[0] + random.gauss(0, 0.4),
                p[1] + random.gauss(0, 0.4),
                p[2] + random.gauss(0, 20.0)
            ])
        inits.append(perturbed)
    
    # Also perturb the ring
    for _ in range(2):
        base = make_ring_initial()
        perturbed = []
        for p in base:
            perturbed.append([
                p[0] + random.gauss(0, 0.3),
                p[1] + random.gauss(0, 0.3),
                p[2] + random.gauss(0, 15.0)
            ])
        inits.append(perturbed)
    
    best_score = float('inf')
    best_cfg = None
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    # Phase 1: L-BFGS optimization from each initial config
    n_inits = len(inits)
    time_per_init = (TIME_LIMIT - 25.0) / n_inits  # reserve 25s for final polish
    
    for idx, init in enumerate(inits):
        if time.time() - t0 > TIME_LIMIT - 25.0:
            break
        
        v0 = flatten(init, [0.0, 0.0], 0.0)
        remaining = TIME_LIMIT - 25.0 - (time.time() - t0)
        if remaining < 2.0:
            break
        
        try:
            # Use L-BFGS-B with smooth objective
            result = minimize(
                _smooth_objective,
                v0,
                method='L-BFGS-B',
                args=(0.0, 50.0),
                options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-14, 'gtol': 1e-9}
            )
            cfg_opt, center_opt, angle_opt = unflatten(result.x)
            
            # Check exact validity
            s = full_score(cfg_opt, tuple(center_opt), angle_opt)
            if s is not None and s < best_score:
                best_score = s
                best_cfg = [row[:] for row in cfg_opt]
                best_center = center_opt[:]
                best_angle = angle_opt
        except Exception:
            pass
    
    if best_cfg is None:
        # Fallback
        best_cfg = make_hex_lattice()
        best_score = full_score(best_cfg, (0.0, 0.0), 0.0)
        if best_score is None:
            best_score = 10.0
        best_center = [0.0, 0.0]
        best_angle = 0.0
    
    # Phase 2: Nelder-Mead polish on the best solution
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 5.0:
        try:
            cfg_polished, center_polished, angle_polished, polished_score = _nelder_mead_polish(
                best_cfg, best_center, best_angle, remaining_time - 2.0
            )
            if polished_score < best_score:
                best_score = polished_score
                best_cfg = cfg_polished
                best_center = center_polished
                best_angle = angle_polished
        except Exception:
            pass
    
    # Phase 3: Another L-BFGS pass with tighter tolerances
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 3.0:
        v_best = flatten(best_cfg, best_center, best_angle)
        try:
            result = minimize(
                _smooth_objective,
                v_best,
                method='L-BFGS-B',
                args=(0.0, 50.0),
                options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            cfg_final, center_final, angle_final = unflatten(result.x)
            final_score = full_score(cfg_final, tuple(center_final), angle_final)
            if final_score is not None and final_score < best_score:
                best_score = final_score
                best_cfg = cfg_final
                best_center = center_final
                best_angle = angle_final
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead polish
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 3.0:
        try:
            cfg_final2, center_final2, angle_final2, final_score2 = _nelder_mead_polish(
                best_cfg, best_center, best_angle, remaining_time - 1.0
            )
            if final_score2 < best_score:
                best_score = final_score2
                best_cfg = cfg_final2
                best_center = center_final2
                best_angle = angle_final2
        except Exception:
            pass
    
    # Time check: if we're running out of time, return what we have
    if time.time() - t0 > TIME_LIMIT - 2.0:
        pass  # Proceed to return
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
