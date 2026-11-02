import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 20.0
EPS = 1e-9

def time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side length and rotation."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def hex_apothem_normals(angle_deg):
    """Return 6 inward unit normals of a regular hexagon (pointing toward center)."""
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        # Outward normal direction for edge k
        theta = a + k * math.pi / 3.0 + math.pi / 6.0
        normals.append((math.cos(theta), math.sin(theta)))
    return normals

def point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    a = math.radians(angle_deg)
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    # Rotate by -a
    ca = math.cos(-a)
    sa = math.sin(-a)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    # In local frame, hexagon has vertices at angle k*pi/3
    # The hexagon is defined by 6 half-planes
    for k in range(6):
        theta = k * math.pi / 3.0 + math.pi / 6.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        # Distance from center along normal must be <= apothem
        dist = lx * nx + ly * ny
        if dist > side * APOTHEM + EPS:
            return False
    return True

def hexagon_distance(cx1, cy1, a1, cx2, cy2, a2, side1=1.0, side2=1.0):
    """
    Compute the minimum distance between two regular hexagons (both side 1).
    Returns the signed distance: negative if overlapping, positive if separated.
    Uses the fact that for two convex polygons, the minimum distance is the
    minimum over all pairs of (edge of poly1, vertex of poly2) and vice versa.
    """
    v1 = hex_vertices(cx1, cy1, side1, a1)
    v2 = hex_vertices(cx2, cy2, side2, a2)
    
    min_dist = float('inf')
    
    # Check if any vertex of poly1 is inside poly2
    for vx, vy in v1:
        if point_in_hex(vx, vy, cx2, cy2, side2, a2):
            return -1.0  # Overlapping
    
    # Check if any vertex of poly2 is inside poly1
    for vx, vy in v2:
        if point_in_hex(vx, vy, cx1, cy1, side1, a1):
            return -1.0  # Overlapping
    
    # Minimum distance between edges
    def seg_seg_dist(p1, p2, p3, p4):
        """Minimum distance between two line segments."""
        d1x = p2[0] - p1[0]
        d1y = p2[1] - p1[1]
        d2x = p4[0] - p3[0]
        d2y = p4[1] - p3[1]
        rx = p1[0] - p3[0]
        ry = p1[1] - p3[1]
        
        a = d1x * d1x + d1y * d1y
        e = d2x * d2x + d2y * d2y
        f = d2x * rx + d2y * ry
        
        if a < EPS and e < EPS:
            return math.sqrt(rx * rx + ry * ry)
        
        if a < EPS:
            t = 0.0
            s = max(0.0, min(1.0, f / e))
            dx = p3[0] + s * d2x - p1[0]
            dy = p3[1] + s * d2y - p1[1]
            return math.sqrt(dx * dx + dy * dy)
        
        c = d1x * rx + d1y * ry
        if e < EPS:
            s = 0.0
            t = max(0.0, min(1.0, -c / a))
            dx = p1[0] + t * d1x - p3[0]
            dy = p1[1] + t * d1y - p3[1]
            return math.sqrt(dx * dx + dy * dy)
        
        b = d1x * d2x + d1y * d2y
        denom = a * e - b * b
        s = 0.0
        t = 0.0
        if denom >= EPS:
            s = max(0.0, min(1.0, (b * f - c * e) / denom))
            t = (b * s + c) / a
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, f / e))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - e) / denom))
        else:
            s = max(0.0, min(1.0, f / e))
        
        dx = p1[0] + t * d1x - (p3[0] + s * d2x)
        dy = p1[1] + t * d1y - (p3[1] + s * d2y)
        return math.sqrt(dx * dx + dy * dy)
    
    for i in range(6):
        for j in range(6):
            e1 = (v1[i], v1[(i+1) % 6])
            e2 = (v2[j], v2[(j+1) % 6])
            d = seg_seg_dist(e1[0], e1[1], e2[0], e2[1])
            if d < min_dist:
                min_dist = d
    
    return min_dist

def smooth_hex_distance(cx1, cy1, a1, cx2, cy2, a2, temperature=0.05):
    """
    Smooth approximation of hexagon distance for optimization.
    Returns a smooth value that is negative when overlapping.
    """
    # Use vertex-edge distance approximation
    v1 = hex_vertices(cx1, cy1, 1.0, a1)
    v2 = hex_vertices(cx2, cy2, 1.0, a2)
    
    min_dist = float('inf')
    
    # For each vertex of hex1, find distance to edges of hex2
    for vx, vy in v1:
        for j in range(6):
            ex1, ey1 = v2[j]
            ex2, ey2 = v2[(j+1) % 6]
            # Distance from point to segment
            dx = ex2 - ex1
            dy = ey2 - ey1
            len_sq = dx * dx + dy * dy
            if len_sq < EPS:
                d = math.sqrt((vx - ex1)**2 + (vy - ey1)**2)
            else:
                t = max(0.0, min(1.0, ((vx - ex1) * dx + (vy - ey1) * dy) / len_sq))
                px = ex1 + t * dx
                py = ey1 + t * dy
                d = math.sqrt((vx - px)**2 + (vy - py)**2)
            if d < min_dist:
                min_dist = d
    
    for vx, vy in v2:
        for i in range(6):
            ex1, ey1 = v1[i]
            ex2, ey2 = v1[(i+1) % 6]
            dx = ex2 - ex1
            dy = ey2 - ey1
            len_sq = dx * dx + dy * dy
            if len_sq < EPS:
                d = math.sqrt((vx - ex1)**2 + (vy - ey1)**2)
            else:
                t = max(0.0, min(1.0, ((vx - ex1) * dx + (vy - ey1) * dy) / len_sq))
                px = ex1 + t * dx
                py = ey1 + t * dy
                d = math.sqrt((vx - px)**2 + (vy - py)**2)
            if d < min_dist:
                min_dist = d
    
    # Check if centers are too close (approximate overlap)
    dx = cx2 - cx1
    dy = cy2 - cy1
    center_dist = math.sqrt(dx * dx + dy * dy)
    if center_dist < 2.0 * APOTHEM - 0.1:
        # Likely overlapping, use smooth penalty
        return -(2.0 * APOTHEM - center_dist) + min_dist * 0.1
    
    return min_dist

def outer_hex_constraint_value(cx, cy, side, angle_deg, inner_pts, inner_angles):
    """
    Check if all inner hexagons fit inside the outer hexagon.
    Returns the minimum "slack" (positive = fits, negative = violates).
    """
    a = math.radians(angle_deg)
    # Outer hexagon inward normals
    for k in range(6):
        theta = a + k * math.pi / 3.0 + math.pi / 6.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        # For each inner hexagon, check all its vertices
        for i in range(N):
            ix, iy, ia = inner_pts[i][0], inner_pts[i][1], inner_pts[i][2]
            verts = hex_vertices(ix, iy, 1.0, ia)
            for vx, vy in verts:
                dist = (vx - cx) * nx + (vy - cy) * ny
                if dist > side * APOTHEM + EPS:
                    return dist - side * APOTHEM
    return float('inf')

def construct():
    if not time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Initial configuration: hexagonal grid
    # Place 11 hexagons in a compact arrangement
    s3 = math.sqrt(3)
    
    # Start with a hexagonal cluster: center + 6 around + 4 more
    init_positions = [
        (0.0, 0.0, 0.0),
        (s3, 0.0, 0.0),
        (s3/2, 1.5, 0.0),
        (-s3/2, 1.5, 0.0),
        (-s3, 0.0, 0.0),
        (-s3/2, -1.5, 0.0),
        (s3/2, -1.5, 0.0),
        (0.0, 3.0, 0.0),
        (s3, 3.0, 0.0),
        (s3/2, 4.5, 0.0),
        (-s3/2, 4.5, 0.0),
    ]
    
    best_side = float('inf')
    best_config = None
    
    # Multiple random restarts
    np.random.seed(42)
    n_restarts = 30
    
    for restart in range(n_restarts):
        if not time_ok():
            break
        
        # Perturb initial positions
        if restart == 0:
            positions = [list(p) for p in init_positions]
            outer_angle = 0.0
        else:
            positions = []
            for px, py, pa in init_positions:
                positions.append([
                    px + np.random.randn() * 0.5,
                    py + np.random.randn() * 0.5,
                    pa + np.random.randn() * 5.0
                ])
            outer_angle = np.random.uniform(0, 30)
        
        # Objective: minimize outer side length
        # Variables: 3*N for inner (x, y, angle) + 3 for outer (cx, cy, side)
        # Outer angle is fixed per restart
        
        def objective(params):
            if not time_ok():
                return 1e6
            
            inner_params = params[:3*N]
            outer_params = params[3*N:]
            cx, cy, side = outer_params
            
            if side < 1.0:
                return 1e6
            
            # Check non-overlap
            for i in range(N):
                for j in range(i+1, N):
                    ix1, iy1, ia1 = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
                    ix2, iy2, ia2 = inner_params[3*j], inner_params[3*j+1], inner_params[3*j+2]
                    d = smooth_hex_distance(ix1, iy1, ia1, ix2, iy2, ia2, temperature=0.03)
                    if d < 0.0:
                        return 1e6 + (-d) * 1000
            
            # Check containment in outer hexagon
            for i in range(N):
                ix, iy, ia = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
                verts = hex_vertices(ix, iy, 1.0, ia)
                for vx, vy in verts:
                    for k in range(6):
                        theta = math.radians(outer_angle) + k * math.pi / 3.0 + math.pi / 6.0
                        nx = math.cos(theta)
                        ny = math.sin(theta)
                        dist = (vx - cx) * nx + (vy - cy) * ny
                        if dist > side * APOTHEM + 1e-6:
                            return 1e6 + (dist - side * APOTHEM) * 1000
            
            return side
        
        # Initial guess for outer center and size
        inner_arr = np.array([[p[0], p[1], p[2]] for p in positions])
        cx_init = np.mean(inner_arr[:, 0])
        cy_init = np.mean(inner_arr[:, 1])
        
        # Estimate initial side
        max_r = 0
        for px, py, pa in positions:
            for vx, vy in hex_vertices(px, py, 1.0, pa):
                r = math.sqrt((vx - cx_init)**2 + (vy - cy_init)**2)
                max_r = max(max_r, r)
        side_init = max_r / APOTHEM + 0.5
        
        x0 = []
        for px, py, pa in positions:
            x0.extend([px, py, pa])
        x0.extend([cx_init, cy_init, side_init])
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.extend([(-10, 10), (-10, 10), (-180, 180)])
        bounds.extend([(-10, 10), (-10, 10), (1.0, 10.0)])
        
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if result.fun < 1e5:
                side = result.fun
                if side < best_side:
                    best_side = side
                    best_config = (result.x[:3*N], outer_angle, result.x[3*N:3*N+2])
                    
        except Exception:
            pass
    
    # If continuous optimization didn't find good result, fall back to lattice
    if best_config is None or best_side > 5.0:
        # Use the lattice approach from the original
        return _lattice_fallback()
    
    inner_params, outer_angle, outer_center = best_config
    cx, cy = outer_center
    
    inner = []
    for i in range(N):
        x = float(inner_params[3*i])
        y = float(inner_params[3*i+1])
        a = float(inner_params[3*i+2]) % 360.0
        inner.append([x, y, a])
    
    # Verify and compute exact side
    max_proj = 0
    for i in range(N):
        x, y, a = inner[i]
        verts = hex_vertices(x, y, 1.0, a)
        for vx, vy in verts:
            for k in range(6):
                theta = math.radians(outer_angle) + k * math.pi / 3.0 + math.pi / 6.0
                nx = math.cos(theta)
                ny = math.sin(theta)
                dist = abs((vx - cx) * nx + (vy - cy) * ny)
                max_proj = max(max_proj, dist)
    
    exact_side = (max_proj + APOTHEM) / APOTHEM
    
    return inner, [float(cx), float(cy)], float(exact_side), float(outer_angle)

def _lattice_fallback():
    """Fallback to lattice-based approach if continuous optimization fails."""
    s3 = math.sqrt(3)
    
    # Best known lattice configuration for 11 hexagons
    axial = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    
    pts = []
    for q, r in axial:
        x = s3 * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    pts = np.array(pts)
    
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    
    # Find optimal rotation
    best_S = float('inf')
    best_beta = 0.0
    
    for i in range(360):
        beta = i * math.pi / 180.0
        max_proj = 0
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        if S < best_S:
            best_S = S
            best_beta = beta
    
    # Refine
    for _ in range(100):
        best_S_local = best_S
        best_beta_local = best_beta
        for db in [-0.01, -0.005, 0.0, 0.005, 0.01]:
            beta = best_beta + db
            max_proj = 0
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                nx, ny = math.cos(ang), math.sin(ang)
                projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            S = (max_proj + APOTHEM) / APOTHEM
            if S < best_S_local:
                best_S_local = S
                best_beta_local = beta
        if abs(best_S_local - best_S) < 1e-10:
            break
        best_S = best_S_local
        best_beta = best_beta_local
    
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), float(math.degrees(best_beta))])
    
    return inner, [cx, cy], float(best_S), float(math.degrees(best_beta))
