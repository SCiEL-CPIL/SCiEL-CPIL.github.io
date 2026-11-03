"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random

N = 12
TIME_LIMIT = 115.0  # safety margin below 120s

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
        # Edge direction
        dx, dy = x2 - x1, y2 - y1
        # Normal
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
            return False  # Found separating axis -> no overlap
    return True  # No separating axis -> overlap

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    # 6 outward normals of the outer hexagon
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

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_initial():
        """Hexagonal lattice arrangement: rows of 4, 4, 4 with offset."""
        pts = []
        # 3 rows x 4 columns, hexagonal packing
        row_height = math.sqrt(3) * 1.0  # vertical spacing for unit hexagons
        col_width = 2.0 * 1.0  # horizontal spacing
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts

    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, center, angle)

    cfg = make_initial()
    center = [0.0, 0.0]
    angle = 0.0
    
    # Try to get a valid initial score
    cur_score = full_score(cfg, tuple(center), angle)
    if cur_score is None:
        # Spread them out more
        for p in cfg:
            p[0] *= 1.5
            p[1] *= 1.5
        cur_score = full_score(cfg, tuple(center), angle)
        if cur_score is None:
            cur_score = 40.0
            # Force validity by spreading
            for i, p in enumerate(cfg):
                p[0] = (i % 4 - 1.5) * 3.0
                p[1] = (i // 4 - 1.0) * 3.0
    
    best_score = cur_score
    best_cfg = [row[:] for row in cfg]
    best_center = center[:]
    best_angle = angle

    T = 1.0
    T_end = 0.005
    alpha = 0.9998
    
    while time.time() - t0 < TIME_LIMIT:
        r = random.random()
        if r < 0.55:
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            cfg[idx][0] += random.gauss(0, 0.25)
            cfg[idx][1] += random.gauss(0, 0.25)
        elif r < 0.75:
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 8.0)
        elif r < 0.85:
            old_cx, old_cy = center[0], center[1]
            center[0] += random.gauss(0, 0.15)
            center[1] += random.gauss(0, 0.15)
        else:
            old_angle = angle
            angle += random.gauss(0, 3.0)
        
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
                # Revert
                if r < 0.55:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.75:
                    cfg[idx][2] = old_ang
                elif r < 0.85:
                    center[0], center[1] = old_cx, old_cy
                else:
                    angle = old_angle
        else:
            # Invalid: revert
            if r < 0.55:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.75:
                cfg[idx][2] = old_ang
            elif r < 0.85:
                center[0], center[1] = old_cx, old_cy
            else:
                angle = old_angle
        
        T = max(T_end, T * alpha)
    
    inner = [[p[0], p[1], p[2]] for p in best_cfg]
    return inner, (best_center[0], best_center[1]), best_score, best_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
