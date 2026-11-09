"""Hexagon packing: place 11 unit hexagons in a minimal regular hexagon."""
import random
import math
import time

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_verts(cx, cy, angle_deg, r=1.0):
    """Vertices of a regular hexagon (circumradius r) centered at (cx,cy)."""
    a = math.radians(angle_deg)
    return [(cx + r * math.cos(a + k * math.pi / 3.0),
             cy + r * math.sin(a + k * math.pi / 3.0)) for k in range(6)]

def _pt_in_poly(px, py, poly):
    """Ray-casting point in convex polygon."""
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

def _seg_seg_intersect(p1, p2, p3, p4):
    """Check if two segments properly intersect."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _pt_seg_closest(px, py, a, b):
    """Closest point on segment ab to point p."""
    dx, dy = b[0]-a[0], b[1]-a[1]
    d2 = dx*dx + dy*dy
    if d2 < 1e-15:
        return a[0], a[1]
    t = max(0.0, min(1.0, ((px-a[0])*dx + (py-a[1])*dy) / d2))
    return a[0]+t*dx, a[1]+t*dy

def _poly_distance(v1, v2):
    """
    Compute minimum distance between two convex polygons.
    Returns (dist, nx, ny) where dist < 0 means overlap,
    and (nx, ny) is the push direction to separate v1 from v2.
    """
    n1, n2 = len(v1), len(v2)
    
    # Check if any vertex of v1 is inside v2
    for i in range(n1):
        if _pt_in_poly(v1[i][0], v1[i][1], v2):
            min_d = float('inf')
            px, py = 0, 0
            for j in range(n2):
                cx, cy = _pt_seg_closest(v1[i][0], v1[i][1], v2[j], v2[(j+1)%n2])
                d = math.sqrt((v1[i][0]-cx)**2 + (v1[i][1]-cy)**2)
                if d < min_d:
                    min_d = d
                    px, py = cx, cy
            dx, dy = v1[i][0]-px, v1[i][1]-py
            ln = math.sqrt(dx*dx+dy*dy)
            if ln > 1e-12:
                return (-min_d, dx/ln, dy/ln)
            return (-min_d, 0.0, 0.0)
    
    # Check if any vertex of v2 is inside v1
    for j in range(n2):
        if _pt_in_poly(v2[j][0], v2[j][1], v1):
            min_d = float('inf')
            px, py = 0, 0
            for i in range(n1):
                cx, cy = _pt_seg_closest(v2[j][0], v2[j][1], v1[i], v1[(i+1)%n1])
                d = math.sqrt((v2[j][0]-cx)**2 + (v2[j][1]-cy)**2)
                if d < min_d:
                    min_d = d
                    px, py = cx, cy
            dx, dy = v2[j][0]-px, v2[j][1]-py
            ln = math.sqrt(dx*dx+dy*dy)
            if ln > 1e-12:
                return (-min_d, -dx/ln, -dy/ln)
            return (-min_d, 0.0, 0.0)
    
    # Check edge intersections
    for i in range(n1):
        for j in range(n2):
            if _seg_seg_intersect(v1[i], v1[(i+1)%n1], v2[j], v2[(j+1)%n2]):
                return (-1.0, 0.0, 0.0)
    
    # Find minimum distance between edges
    min_d = float('inf')
    min_nx, min_ny = 0.0, 0.0
    for i in range(n1):
        a1, a2 = v1[i], v1[(i+1)%n1]
        for j in range(n2):
            b1, b2 = v2[j], v2[(j+1)%n2]
            # Check all 4 point-to-segment distances
            for (px, py, seg_a, seg_b) in [(a1[0],a1[1],b1,b2), (a2[0],a2[1],b1,b2),
                                            (b1[0],b1[1],a1,a2), (b2[0],b2[1],a1,a2)]:
                cx, cy = _pt_seg_closest(px, py, seg_a, seg_b)
                d = math.sqrt((px-cx)**2 + (py-cy)**2)
                if d < min_d:
                    min_d = d
                    # Normal pointing from v1 to v2
                    m1x, m1y = (a1[0]+a2[0])/2, (a1[1]+a2[1])/2
                    m2x, m2y = (b1[0]+b2[0])/2, (b1[1]+b2[1])/2
                    dx, dy = m2x-m1x, m2y-m1y
                    ln = math.sqrt(dx*dx+dy*dy)
                    if ln > 1e-12:
                        min_nx, min_ny = dx/ln, dy/ln
    return (min_d, min_nx, min_ny)

def _max_radius_to_hex(cx, cy, angle, outer_side):
    """
    Maximum circumradius of a hexagon at (cx,cy,angle) that fits inside
    outer hexagon of given side centered at origin.
    Returns the maximum scale factor (1.0 means it fits as-is).
    """
    # Outer hex vertices
    outer = [(outer_side * math.cos(k*math.pi/3), outer_side * math.sin(k*math.pi/3)) for k in range(6)]
    
    # Inner hex vertices at unit scale
    inner_unit = _hex_verts(0, 0, angle, 1.0)
    inner_translated = [(cx + v[0], cy + v[1]) for v in inner_unit]
    
    # Binary search on scale
    lo, hi = 0.1, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        inner_scaled = [(cx + v[0]*mid, cy + v[1]*mid) for v in inner_unit]
        fits = True
        for vx, vy in inner_scaled:
            if not _pt_in_poly(vx, vy, outer):
                fits = False
                break
        if fits:
            lo = mid
        else:
            hi = mid
    return lo

def _build_cluster_config(theta, phi, gap_angle, positions_11):
    """
    Build a contact-graph configuration.
    - Central hex at origin
    - 6 ring hexes at distance 2*sqrt(3) from center (touching central)
    - 4 additional hexes in gaps
    theta: rotation of entire cluster
    phi: angular offset of ring
    gap_angle: angle for gap hexes
    positions_11: list of 11 (dx, dy, angle_deg) relative positions
    """
    return positions_11

def _min_enclosing_hex(hexes):
    """
    Find minimum side length of a regular hexagon (any rotation, any center)
    that contains all given hexagons.
    Uses optimization over center and rotation.
    """
    def _contains_all(cx, cy, rot_deg, side):
        outer = [(cx + side*math.cos(math.radians(rot_deg) + k*math.pi/3),
                  cy + side*math.sin(math.radians(rot_deg) + k*math.pi/3)) for k in range(6)]
        for hx, hy, ha in hexes:
            verts = _hex_verts(hx, hy, ha, 1.0)
            for vx, vy in verts:
                if not _pt_in_poly(vx, vy, outer):
                    return False
        return True
    
    def _required_side(cx, cy, rot_deg):
        lo, hi = 2.0, 15.0
        for _ in range(35):
            mid = (lo + hi) / 2
            if _contains_all(cx, cy, rot_deg, mid):
                hi = mid
            else:
                lo = mid
        return hi
    
    # Optimize over center and rotation
    best_side = float('inf')
    best_params = (0, 0, 0)
    
    # Try multiple initial guesses for center and rotation
    random.seed(123)
    for _ in range(50):
        # Estimate center as centroid of all hex centers
        cx0 = sum(h[0] for h in hexes) / N
        cy0 = sum(h[1] for h in hexes) / N
        rot0 = random.uniform(0, 60)
        
        cx, cy, rot = cx0, cy0, rot0
        current_side = _required_side(cx, cy, rot)
        
        # Local optimization
        step = 0.5
        for _ in range(30):
            improved = False
            for _ in range(10):
                dcx = random.uniform(-step, step)
                dcy = random.uniform(-step, step)
                drot = random.uniform(-step, step)
                ns = _required_side(cx+dcx, cy+dcy, rot+drot)
                if ns < current_side:
                    cx, cy, rot = cx+dcx, cy+dcy, rot+drot
                    current_side = ns
                    improved = True
            if not improved:
                step *= 0.7
                if step < 0.01:
                    break
        
        if current_side < best_side:
            best_side = current_side
            best_params = (cx, cy, rot)
    
    return best_side, best_params

def _validate_config(hexes, side, ox, oy, orot):
    """Check if configuration is valid: no overlaps, all inside outer hex."""
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            v1 = _hex_verts(hexes[i][0], hexes[i][1], hexes[i][2], 1.0)
            v2 = _hex_verts(hexes[j][0], hexes[j][1], hexes[j][2], 1.0)
            d, _, _ = _poly_distance(v1, v2)
            if d < -1e-8:
                return False
    
    # Check containment
    outer = [(ox + side*math.cos(math.radians(orot) + k*math.pi/3),
              oy + side*math.sin(math.radians(orot) + k*math.pi/3)) for k in range(6)]
    for hx, hy, ha in hexes:
        verts = _hex_verts(hx, hy, ha, 1.0)
        for vx, vy in verts:
            if not _pt_in_poly(vx, vy, outer):
                return False
    return True

def construct():
    t0 = time.time()
    
    # Contact graph: central hex + 6 ring + 4 gap fillers
    # For regular hexagons of side 1:
    # - Distance between centers of touching hexes = 2 * inradius = 2 * (sqrt(3)/2) = sqrt(3)
    # Wait, for regular hexagons of side s:
    #   inradius = s * sqrt(3)/2
    #   Distance between centers of two touching hexagons = 2 * inradius = s * sqrt(3)
    # For s=1: distance = sqrt(3) ≈ 1.732
    
    sqrt3 = math.sqrt(3)
    d_touch = sqrt3  # center-to-center distance for touching hexagons
    
    # Generate multiple cluster topologies
    clusters = []
    
    # Topology 1: Central + 6 ring (hexagonal flower) + 4 in outer gaps
    # Ring hexes at angles 0, 60, 120, 180, 240, 300 degrees from center
    # Gap hexes at angles 30, 90, 150, 210, 270, 330 (but we only need 4)
    
    for cluster_rot in [0, 30, 45, 60, 90, 120]:
        for gap_subset in range(4):  # Different choices of which 4 gaps to fill
            hexes = []
            # Central hex at origin
            hexes.append([0.0, 0.0, 0.0])
            
            # 6 ring hexes
            for k in range(6):
                angle = math.radians(cluster_rot + k * 60)
                rx = d_touch * math.cos(angle)
                ry = d_touch * math.sin(angle)
                hexes.append([rx, ry, 0.0])
            
            # 4 gap hexes: choose 4 from the 6 gap positions
            gap_angles = [cluster_rot + 30 + k * 60 for k in range(6)]
            # Select 4 gaps based on gap_subset
            if gap_subset == 0:
                selected_gaps = [0, 1, 3, 4]
            elif gap_subset == 1:
                selected_gaps = [0, 2, 3, 5]
            elif gap_subset == 2:
                selected_gaps = [1, 2, 4, 5]
            else:
                selected_gaps = [0, 1, 2, 4]
            
            for gi in selected_gaps:
                angle = math.radians(gap_angles[gi])
                # Gap hexes are at distance d_touch * sqrt(3) from center (touching two ring hexes)
                # Actually, for a hex between two ring hexes:
                # The gap position is at distance d_touch * sqrt(3) from center? No.
                # Let me think: ring hexes are at distance d_touch from center.
                # A gap hex touching two adjacent ring hexes would be at:
                # If ring hexes are at angles a and a+60, the gap is at angle a+30
                # and at distance d_touch * sqrt(3) from center? 
                # Actually: distance from center to gap = d_touch * sqrt(3)
                # Because the three centers form an equilateral triangle with side d_touch,
                # and the gap center is at distance d_touch/sqrt(3) * sqrt(3) = d_touch from each ring center...
                # Wait, let me recalculate.
                # Central at (0,0), ring at (d_touch, 0) and (d_touch*cos60, d_touch*sin60)
                # = (d_touch/2, d_touch*sqrt(3)/2)
                # Gap hex touches both ring hexes, so it's at distance d_touch from each.
                # By symmetry, it's on the angle bisector at 30 degrees.
                # Distance from center: let's call it r.
                # Distance from gap to ring1: sqrt(r^2 + d_touch^2 - 2*r*d_touch*cos(30)) = d_touch
                # r^2 + d_touch^2 - 2*r*d_touch*(sqrt(3)/2) = d_touch^2
                # r^2 - r*d_touch*sqrt(3) = 0
                # r*(r - d_touch*sqrt(3)) = 0
                # r = d_touch*sqrt(3)
                gap_dist = d_touch * sqrt3
                gx = gap_dist * math.cos(angle)
                gy = gap_dist * math.sin(angle)
                hexes.append([gx, gy, 0.0])
            
            clusters.append(hexes)
    
    # Topology 2: Linear strip with branches
    for strip_offset in [0, 1]:
        hexes = []
        # 5 in a row
        for i in range(5):
            x = (i - 2) * d_touch
            hexes.append([x, 0.0, 0.0])
        # 3 above, 3 below (staggered)
        for i in range(3):
            x = (i - 1) * d_touch * 1.5
            hexes.append([x, d_touch * sqrt3/2, 0.0])
        for i in range(3):
            x = (i - 1) * d_touch * 1.5
            hexes.append([x, -d_touch * sqrt3/2, 0.0])
        clusters.append(hexes)
    
    # Topology 3: 3-4-4 grid (original but with proper touching distances)
    hexes = []
    # Row 0: 3 hexes
    for i in range(3):
        x = (i - 1) * d_touch
        hexes.append([x, 0.0, 0.0])
    # Row 1: 4 hexes, offset
    for i in range(4):
        x = (i - 1.5) * d_touch
        hexes.append([x, d_touch * sqrt3/2, 0.0])
    # Row 2: 4 hexes
    for i in range(4):
        x = (i - 1.5) * d_touch
        hexes.append([x, d_touch * sqrt3, 0.0])
    clusters.append(hexes)
    
    # For each cluster, try to optimize
    best_side = float('inf')
    best_result = None
    
    for ci, hexes in enumerate(clusters):
        if time.time() - t0 > DEADLINE - 15:
            break
        
        # Verify no overlaps in this cluster
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                v1 = _hex_verts(hexes[i][0], hexes[i][1], hexes[i][2], 1.0)
                v2 = _hex_verts(hexes[j][0], hexes[j][1], hexes[j][2], 1.0)
                d, _, _ = _poly_distance(v1, v2)
                if d < -1e-8:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Find minimum enclosing hexagon
        side, (ox, oy, orot) = _min_enclosing_hex(hexes)
        
        if side < best_side:
            best_side = side
            best_result = (hexes, ox, oy, orot)
    
    # If no valid cluster found, fall back to simple grid
    if best_result is None:
        hexes = []
        for i in range(3):
            hexes.append([(i-1)*2.0, 0.0, 0.0])
        for i in range(4):
            hexes.append([(i-1.5)*2.0, 1.8, 0.0])
        for i in range(4):
            hexes.append([(i-1.5)*2.0, 3.6, 0.0])
        side, (ox, oy, orot) = _min_enclosing_hex(hexes)
        best_result = (hexes, ox, oy, orot)
        best_side = side
    
    hexes, ox, oy, orot = best_result
    
    # Final validation
    if not _validate_config(hexes, best_side, ox, oy, orot):
        # Expand side slightly
        best_side *= 1.01
    
    inner = [[hexes[i][0], hexes[i][1], hexes[i][2]] for i in range(N)]
    return inner, (ox, oy), best_side, orot

