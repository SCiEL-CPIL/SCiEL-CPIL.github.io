import numpy as np
import time
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
DEADLINE = 115.0  # 5s safety margin

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side, rotation."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle = np.radians(angle_deg)
    # Rotate point by -angle
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local frame, hexagon vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is the intersection of 6 half-planes
    # For a unit hexagon (side=1), the inradius is sqrt(3)/2
    # The constraint: for each edge normal direction, |projection| <= inradius
    # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees (perpendicular to edges)
    for i in range(6):
        normal_angle = np.radians(30 + i * 60)
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        proj = lx * nx + ly * ny
        if proj > side * np.sqrt(3) / 2 + 1e-10:
            return False
    return True

def min_center_distance():
    """Minimum center-to-center distance for two non-overlapping unit hexagons."""
    return np.sqrt(3)

def check_feasible(inner, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons are inside outer and don't overlap."""
    ocx, ocy = outer_center
    # Check containment
    for (ix, iy, iang) in inner:
        # Check all 6 vertices of inner hexagon are inside outer
        verts = hexagon_vertices(ix, iy, 1.0, iang)
        for v in verts:
            if not point_in_hexagon(v[0], v[1], ocx, ocy, outer_side, outer_angle):
                return False
    # Check pairwise non-overlap using conservative distance constraint
    min_dist = min_center_distance()
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            dx = inner[i][0] - inner[j][0]
            dy = inner[i][1] - inner[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_dist - 1e-10:
                return False
    return True

def objective(params, outer_center, outer_side, outer_angle):
    """Objective: maximize minimum margin (or minimize violation)."""
    inner = []
    for i in range(N):
        ix = params[3*i]
        iy = params[3*i+1]
        iang = params[3*i+2]
        inner.append([ix, iy, iang])
    
    # Compute minimum containment margin
    min_margin = np.inf
    ocx, ocy = outer_center
    for (ix, iy, iang) in inner:
        verts = hexagon_vertices(ix, iy, 1.0, iang)
        for v in verts:
            # Compute margin: how far inside the outer hexagon
            dx = v[0] - ocx
            dy = v[1] - ocy
            angle = np.radians(outer_angle)
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            # The inradius of outer hexagon is outer_side * sqrt(3)/2
            inradius_outer = outer_side * np.sqrt(3) / 2
            for i in range(6):
                normal_angle = np.radians(30 + i * 60)
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                proj = lx * nx + ly * ny
                margin = inradius_outer - proj
                if margin < min_margin:
                    min_margin = margin
    
    # Compute minimum separation
    min_sep = np.inf
    min_dist = min_center_distance()
    for i in range(N):
        for j in range(i+1, N):
            dx = inner[i][0] - inner[j][0]
            dy = inner[i][1] - inner[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            sep = dist - min_dist
            if sep < min_sep:
                min_sep = sep
    
    # We want both min_margin > 0 and min_sep > 0
    # Use a penalty approach: return negative of the minimum of these
    # If either is negative, we have a violation
    return -min(min_margin, min_sep)

def initial_guess():
    """Generate a good initial arrangement."""
    # Hexagonal lattice arrangement
    # Place hexagons in a hexagonal pattern
    # For 12 hexagons, try a 3-4-3-2 or similar pattern
    # Use spacing of sqrt(3) for center-to-center
    spacing = np.sqrt(3)
    positions = []
    
    # Try a compact hexagonal arrangement
    # Center at origin, then rings
    # Ring 0: 1 hex
    positions.append([0.0, 0.0, 0.0])
    # Ring 1: 6 hexes at distance sqrt(3)
    for i in range(6):
        angle = i * np.pi / 3
        x = spacing * np.cos(angle)
        y = spacing * np.sin(angle)
        positions.append([x, y, 0.0])
    # That's 7. Need 5 more. Add them in a second partial ring
    # Place 5 more at distance 2*sqrt(3) or in gaps
    # Actually, let's try a different arrangement
    # 4-4-4 rows or 3-4-3-2
    
    # Let's use a rectangular-ish grid with hexagonal offset
    # Row 0: 4 hexes
    # Row 1: 4 hexes (offset)
    # Row 2: 4 hexes
    positions = []
    row_spacing = 1.5  # vertical spacing for hex grid
    col_spacing = spacing  # horizontal
    
    for row in range(3):
        for col in range(4):
            x = col * col_spacing
            if row % 2 == 1:
                x += col_spacing / 2
            y = row * row_spacing
            positions.append([x, y, 0.0])
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    positions = [[p[0]-cx, p[1]-cy, p[2]] for p in positions]
    
    return positions

def construct():
    deadline = time.time() + DEADLINE - (time.time() - START_TIME)
    if deadline <= 0:
        # Return a simple valid solution
        inner = initial_guess()
        # Estimate required outer side
        xs = [p[0] for p in inner]
        ys = [p[1] for p in inner]
        max_r = max(np.sqrt(x*x + y*y) for x, y in zip(xs, ys)) + 1.0
        outer_side = max_r * 2 / np.sqrt(3) + 1.0
        return inner, (0.0, 0.0), outer_side, 0.0
    
    # Start with a good initial guess
    inner = initial_guess()
    
    # Estimate initial outer side
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    max_r = max(np.sqrt(x*x + y*y) for x, y in zip(xs, ys)) + 1.0
    outer_side = max_r * 2 / np.sqrt(3) + 0.5
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Optimize: fix outer hexagon, optimize inner positions
    # Or optimize both. Let's optimize inner positions with a fixed outer, then adjust outer.
    
    # Build parameter vector
    params = np.array([p for pos in inner for p in pos])
    
    # Use Nelder-Mead or L-BFGS-B
    best_params = params.copy()
    best_obj = objective(params, outer_center, outer_side, outer_angle)
    
    # Multi-start local optimization
    n_starts = 5
    for start in range(n_starts):
        if time.time() > deadline:
            break
        
        if start == 0:
            p = params.copy()
        else:
            p = params + np.random.randn(3*N) * 0.1
        
        # Optimize
        try:
            res = minimize(objective, p, args=(outer_center, outer_side, outer_angle),
                          method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except:
            pass
    
    inner_opt = []
    for i in range(N):
        inner_opt.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    # Now find the minimal outer hexagon that contains all inner hexagons
    # For a given inner configuration, find min outer side
    # The outer hexagon should be centered and rotated optimally
    
    def compute_min_outer_side(inner_cfg, outer_angle=0.0):
        """Compute minimum outer side length for given inner config and outer angle."""
        # The outer hexagon center should be at the centroid of inner hexagon centers
        xs = [p[0] for p in inner_cfg]
        ys = [p[1] for p in inner_cfg]
        ocx = sum(xs) / len(xs)
        ocy = sum(ys) / len(ys)
        
        # For a hexagon with center (ocx, ocy), side S, angle a,
        # the constraint is that all vertices of all inner hexagons are inside
        # This is a linear constraint in S for fixed angle
        # min S such that for all inner vertices v:
        #   for all 6 edge normals n_i: dot(v - c, n_i) <= S * sqrt(3)/2
        
        min_S = 0.0
        for (ix, iy, iang) in inner_cfg:
            verts = hexagon_vertices(ix, iy, 1.0, iang)
            for v in verts:
                dx = v[0] - ocx
                dy = v[1] - ocy
                angle = np.radians(outer_angle)
                cos_a = np.cos(-angle)
                sin_a = np.sin(-angle)
                lx = dx * cos_a - dy * sin_a
                ly = dx * sin_a + dy * cos_a
                
                for i in range(6):
                    normal_angle = np.radians(30 + i * 60)
                    nx = np.cos(normal_angle)
                    ny = np.sin(normal_angle)
                    proj = lx * nx + ly * ny
                    # proj <= S * sqrt(3)/2  =>  S >= proj * 2/sqrt(3)
                    if proj > 0:
                        S_needed = proj * 2 / np.sqrt(3)
                        if S_needed > min_S:
                            min_S = S_needed
        
        return min_S, (ocx, ocy)
    
    # Optimize outer angle
    best_side = np.inf
    best_outer_angle = 0.0
    best_oc = (0.0, 0.0)
    
    for ang in range(0, 60, 5):
        if time.time() > deadline:
            break
        side, oc = compute_min_outer_side(inner_opt, float(ang))
        if side < best_side:
            best_side = side
            best_outer_angle = float(ang)
            best_oc = oc
    
    # Fine-tune outer angle
    for _ in range(3):
        best_side_local = best_side
        best_ang_local = best_outer_angle
        best_oc_local = best_oc
        for da in [-2, -1, -0.5, 0.5, 1, 2]:
            ang = best_outer_angle + da
            side, oc = compute_min_outer_side(inner_opt, ang)
            if side < best_side_local:
                best_side_local = side
                best_ang_local = ang
                best_oc_local = oc
        if best_side_local >= best_side:
            break
        best_side = best_side_local
        best_outer_angle = best_ang_local
        best_oc = best_oc_local
    
    # Verify feasibility
    if not check_feasible(inner_opt, best_oc, best_side, best_outer_angle):
        # Fallback: use a larger outer
        inner = initial_guess()
        xs = [p[0] for p in inner]
        ys = [p[1] for p in inner]
        max_r = max(np.sqrt(x*x + y*y) for x, y in zip(xs, ys)) + 1.0
        outer_side = max_r * 2 / np.sqrt(3) + 1.0
        return inner, (0.0, 0.0), outer_side, 0.0
    
    return inner_opt, best_oc, best_side, best_outer_angle
