"""Force-directed packing of 11 unit hexagons inside a regular hexagon."""
import time
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, r, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with circumradius r, rotated by angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return verts

def _point_in_polygon(px, py, poly):
    """Ray casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _seg_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    """Check if segment (a,b) intersects segment (c,d). Returns True if they cross (excluding shared endpoints)."""
    def orient(px, py, qx, qy, rx, ry):
        val = (qx - px) * (ry - py) - (qy - py) * (rx - px)
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else -1

    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)

    if o1 != o2 and o3 != o4:
        return True
    return False

def _polygons_overlap(poly1, poly2):
    """Check if two convex polygons overlap using SAT-like approach: check if any vertex is inside the other, or if edges cross."""
    # Check if any vertex of poly1 is inside poly2
    for vx, vy in poly1:
        if _point_in_polygon(vx, vy, poly2):
            return True
    # Check if any vertex of poly2 is inside poly1
    for vx, vy in poly2:
        if _point_in_polygon(vx, vy, poly1):
            return True
    # Check edge crossings
    n1 = len(poly1)
    n2 = len(poly2)
    for i in range(n1):
        ax, ay = poly1[i]
        bx, by = poly1[(i + 1) % n1]
        for j in range(n2):
            cx, cy = poly2[j]
            dx, dy = poly2[(j + 1) % n2]
            if _seg_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                return True
    return False

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of outer regular hexagon with side length 'side', centered at (cx,cy), rotated by angle_deg."""
    # For a regular hexagon with side length s, the circumradius is s
    r = side
    verts = []
    a = math.radians(angle_deg)
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return verts

def _inner_fully_inside(inner_poly, outer_poly):
    """Check if all vertices of inner_poly are inside outer_poly (sufficient for convex polygons)."""
    for vx, vy in inner_poly:
        if not _point_in_polygon(vx, vy, outer_poly):
            return False
    return True

def _is_valid_configuration(inner, center, side, angle):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    cx, cy = center
    outer = _outer_hex_vertices(cx, cy, side, angle)
    inner_polys = []
    for (ix, iy, iang) in inner:
        inner_polys.append(_hex_vertices(ix, iy, 1.0, iang))
    
    # Check all inner hexagons are inside outer
    for ip in inner_polys:
        if not _inner_fully_inside(ip, outer):
            return False
    
    # Check no two inner hexagons overlap
    for i in range(len(inner_polys)):
        for j in range(i + 1, len(inner_polys)):
            if _polygons_overlap(inner_polys[i], inner_polys[j]):
                return False
    return True

def _compute_min_side(inner, center, angle):
    """Given inner positions and outer center/angle, compute minimum outer side length that contains all inner hexagons."""
    cx, cy = center
    # For a regular hexagon with side s and angle, the support function in direction theta is s * max over vertices of dot product
    # We need to find min s such that all inner vertices are inside.
    # For a regular hexagon centered at (cx,cy) with circumradius s and rotation angle, 
    # a point p is inside if for all 6 edge normals n_k, dot(p - center, n_k) <= s * h_k
    # where h_k is the apothem = s * sqrt(3)/2
    # Actually, let's use the vertex containment: all 6*11 inner vertices must be inside the outer hex.
    # The outer hex has vertices at distance s from center.
    # A point (px,py) is inside the outer hex (centered at (cx,cy), rotation angle) iff:
    # Let d = (px-cx, py-cy). Rotate d by -angle. Then |d_rot|_hex <= s where |.|_hex is the hexagonal norm.
    # The hexagonal norm of a vector (u,v) is max(|u|, |v|, |u+v|) / (sqrt(3)/2) ... let me think more carefully.
    
    # For a regular hexagon with flat-top orientation (angle=0), vertices at angles 0,60,120,180,240,300 degrees, circumradius s.
    # The hexagon is the set of points where max(|x|, |y|, |x+y|) <= s * sqrt(3)/2? No...
    # Let me just use the vertex approach: the outer hex contains a point if the point is on the same side of each edge as the center.
    
    a = math.radians(angle)
    # Edge normals for outer hexagon (pointing inward)
    # Vertices at angle a + k*pi/3 for k=0..5
    # Edge between vertex k and k+1 has normal pointing toward center
    # Let's compute the 6 support values needed
    
    # For each inner vertex, compute the required side length
    max_required = 0.0
    for (ix, iy, iang) in inner:
        ip = _hex_vertices(ix, iy, 1.0, iang)
        for (vx, vy) in ip:
            # Compute distance from (vx,vy) to center in hexagonal metric
            dx = vx - cx
            dy = vy - cy
            # Rotate by -angle
            ca = math.cos(-a)
            sa = math.sin(-a)
            rx = dx * ca - dy * sa
            ry = dx * sa + dy * ca
            # For a regular hexagon with circumradius s (vertices at distance s), 
            # the condition for (rx,ry) to be inside is:
            # The hexagon has edges at distance s*sqrt(3)/2 from center (apothem).
            # The 6 edge normals (for flat-top, angle=0) are at angles 30,90,150,210,270,330 degrees
            # Normal k: (cos(30+k*60), sin(30+k*60))
            # Condition: dot((rx,ry), normal_k) <= s * sqrt(3)/2 for all k
            # So s >= 2 * max_k(dot((rx,ry), normal_k)) / sqrt(3)
            required = 0.0
            for k in range(6):
                nk = math.radians(30 + k * 60)
                dot = rx * math.cos(nk) + ry * math.sin(nk)
                if dot > required:
                    required = dot
            s_needed = required * 2.0 / math.sqrt(3.0)
            if s_needed > max_required:
                max_required = s_needed
    return max_required

def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    best_side = 40.0
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    np.random.seed(42)
    
    # Initial configuration: place hexagons in a rough grid
    # Unit hexagons have circumradius 1, so they span about 2 units
    # Place them in a pattern with spacing ~2.2
    
    def _init_config():
        """Initialize inner hexagons in a compact pattern."""
        configs = []
        # Pattern 1: triangular lattice
        rows = 3
        cols = 4
        pos = []
        for r in range(rows):
            for c in range(cols):
                x = c * 2.1 + (r % 2) * 1.05
                y = r * 1.82
                pos.append((x, y, 0.0))
        # Trim to N
        pos = pos[:N]
        # Center them
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        mx = (min(xs) + max(xs)) / 2
        my = (min(ys) + max(ys)) / 2
        pos = [(x - mx, y - my, a) for (x, y, a) in pos]
        configs.append(pos)
        
        # Pattern 2: random
        pos2 = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            a = np.random.uniform(0, 180)
            pos2.append((x, y, a))
        configs.append(pos2)
        
        # Pattern 3: ring + center
        pos3 = [(0.0, 0.0, 0.0)]
        for i in range(10):
            ang = i * 36.0
            r = 2.2
            pos3.append((r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang)), ang))
        configs.append(pos3)
        
        return configs
    
    all_configs = _init_config()
    
    for init_idx, init_pos in enumerate(all_configs):
        if time.time() > deadline:
            break
        
        # Force-directed simulation
        positions = [list(p) for p in init_pos]  # [x, y, angle]
        
        # Simulation parameters
        dt = 0.01
        max_iter = 2000
        cooling = 1.0
        cool_rate = 0.9995
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute forces
            forces = [[0.0, 0.0, 0.0] for _ in range(N)]
            
            # Repulsion between inner hexagons
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 1e-6:
                        dx = np.random.uniform(-0.1, 0.1)
                        dy = np.random.uniform(-0.1, 0.1)
                        dist = math.sqrt(dx * dx + dy * dy)
                    
                    # Check if they might overlap (distance < 2*circumradius = 2)
                    if dist < 2.5:
                        # Compute actual polygon overlap
                        pi = _hex_vertices(positions[i][0], positions[i][1], 1.0, positions[i][2])
                        pj = _hex_vertices(positions[j][0], positions[j][1], 1.0, positions[j][2])
                        
                        if _polygons_overlap(pi, pj):
                            # Strong repulsive force
                            f = 10.0 / max(dist, 0.1)
                            fx = f * dx / dist
                            fy = f * dy / dist
                            forces[i][0] += fx
                            forces[i][1] += fy
                            forces[j][0] -= fx
                            forces[j][1] -= fy
                            # Also add angular force to help separate
                            forces[i][2] += np.random.uniform(-1, 1)
                            forces[j][2] += np.random.uniform(-1, 1)
                    
                    # Soft repulsion to prevent getting too close
                    if dist < 2.2:
                        f = 2.0 * (2.2 - dist)
                        fx = f * dx / dist
                        fy = f * dy / dist
                        forces[i][0] += fx
                        forces[i][1] += fy
                        forces[j][0] -= fx
                        forces[j][1] -= fy
            
            # Centering force (keep cluster near origin)
            for i in range(N):
                forces[i][0] -= 0.5 * positions[i][0]
                forces[i][1] -= 0.5 * positions[i][1]
                # Damping on angle
                forces[i][2] -= 0.1 * positions[i][2]
            
            # Update positions
            for i in range(N):
                positions[i][0] += dt * forces[i][0] * cooling
                positions[i][1] += dt * forces[i][1] * cooling
                positions[i][2] += dt * forces[i][2] * cooling * 0.1
                # Normalize angle
                positions[i][2] = positions[i][2] % 360.0
            
            cooling *= cool_rate
        
        # After simulation, compute the bounding side length
        inner = [[p[0], p[1], p[2]] for p in positions]
        side = _compute_min_side(inner, (0.0, 0.0), 0.0)
        
        if side < best_side:
            best_side = side
            best_inner = [list(p) for p in inner]
            best_center = (0.0, 0.0)
            best_angle = 0.0
        
        # Try a few different outer angles
        for oa in [0, 30]:
            side_a = _compute_min_side(inner, (0.0, 0.0), oa)
            if side_a < best_side:
                best_side = side_a
                best_inner = [list(p) for p in inner]
                best_center = (0.0, 0.0)
                best_angle = oa
    
    # Final polish: try to shrink by optimizing outer center
    if best_inner is not None:
        for _ in range(50):
            if time.time() > deadline:
                break
            # Try shifting the inner cluster slightly
            improved = False
            for dx in [-0.1, 0, 0.1]:
                for dy in [-0.1, 0, 0.1]:
                    shifted = [[p[0] + dx, p[1] + dy, p[2]] for p in best_inner]
                    s = _compute_min_side(shifted, (0.0, 0.0), best_angle)
                    if s < best_side - 1e-6:
                        best_side = s
                        best_inner = shifted
                        improved = True
            if not improved:
                break
    
    # Validate
    if best_inner is None:
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_side = 40.0
        best_center = (0.0, 0.0)
        best_angle = 0.0
    
    return best_inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END
