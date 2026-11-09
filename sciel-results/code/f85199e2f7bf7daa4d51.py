import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, S, angle_deg):
    """Check if point (px,py) is inside regular hexagon centered at (cx,cy) with side S and rotation angle_deg."""
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle = np.radians(angle_deg)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = cos_a * dx - sin_a * dy
    ly = sin_a * dx + cos_a * dy
    
    # For a regular hexagon with side S, centered at origin, flat-top orientation:
    # The hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # Half-width (flat-to-flat) = S * sqrt(3)
    # Half-height (point-to-point) = S
    # Actually for pointy-top: vertices at 30, 90, 150, 210, 270, 330
    # Let me use the standard: for a hexagon with "radius" R (center to vertex) = S
    # The containment condition: for each of 6 sides, the point must be on the correct side
    
    # Vertices of outer hexagon (local frame, angle 0 means vertex at angle 0)
    R = S  # circumradius = side length for regular hexagon
    v = []
    for i in range(6):
        theta = i * np.pi / 3
        v.append([R * np.cos(theta), R * np.sin(theta)])
    v = np.array(v)
    
    # For each edge, check if point is inside
    for i in range(6):
        p1 = v[i]
        p2 = v[(i + 1) % 6]
        edge = p2 - p1
        # Normal pointing inward
        normal = np.array([-edge[1], edge[0]])
        # Ensure normal points inward (toward center)
        center_to_mid = (p1 + p2) / 2
        if np.dot(normal, center_to_mid) < 0:
            normal = -normal
        # Point must be on the inward side
        vec = np.array([lx, ly]) - p1
        if np.dot(vec, normal) < -1e-10:
            return False
    return True

def max_violation(inner, outer_center, S, outer_angle):
    """Return the maximum distance by which any inner hexagon vertex violates the outer hexagon boundary."""
    cx, cy = outer_center
    max_v = 0.0
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, iang)
        for (vx, vy) in verts:
            if not point_in_hexagon(vx, vy, cx, cy, S, outer_angle):
                # Compute how far outside
                # Transform to local frame
                dx = vx - cx
                dy = vy - cy
                angle = np.radians(outer_angle)
                cos_a = np.cos(-angle)
                sin_a = np.sin(-angle)
                lx = cos_a * dx - sin_a * dy
                ly = sin_a * dx + cos_a * dy
                
                R = S
                v = []
                for i in range(6):
                    theta = i * np.pi / 3
                    v.append([R * np.cos(theta), R * np.sin(theta)])
                v = np.array(v)
                
                # Find maximum violation
                for i in range(6):
                    p1 = v[i]
                    p2 = v[(i + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    center_to_mid = (p1 + p2) / 2
                    if np.dot(normal, center_to_mid) < 0:
                        normal = -normal
                    vec = np.array([lx, ly]) - p1
                    dist = np.dot(vec, normal)
                    if dist < max_v:
                        max_v = dist
    return -max_v  # positive if violation

def construct():
    # Initial arrangement: hexagonal cluster
    # Place 12 hexagons in a compact pattern
    # Use a hexagonal grid: center + surrounding rings
    # For 12: 1 center, 6 in first ring, 5 in second ring (partial)
    
    # Let's use a row-based hexagonal packing
    # Pointy-top hexagons
    # Row spacing: 1.5 (vertical), col spacing: sqrt(3) (horizontal)
    # Odd rows offset by sqrt(3)/2
    
    sqrt3 = np.sqrt(3)
    
    # Pattern: rows with counts [2, 3, 4, 3] = 12
    rows = [2, 3, 4, 3]
    inner = []
    y_offset = 0
    for r, count in enumerate(rows):
        y = r * 1.5
        x_start = -(count - 1) * sqrt3 / 2
        for c in range(count):
            x = x_start + c * sqrt3
            if r % 2 == 1:
                x += sqrt3 / 2
            inner.append([x, y, 0.0])
    
    # Center the arrangement
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx0 = (max(xs) + min(xs)) / 2
    cy0 = (max(ys) + min(ys)) / 2
    inner = [[p[0] - cx0, p[1] - cy0, p[2]] for p in inner]
    
    # Estimate outer side length
    # The bounding box of the arrangement
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    # For a hexagon, side S has width sqrt(3)*S and height 2S (pointy-top)
    S_est = max(width / sqrt3, height / 2, 2.0)
    
    # Optimize: minimize S such that all inner hexagons fit
    # Use differential evolution to find good positions, then compute S
    
    def objective(params):
        # params: x, y, angle for each of 12 hexagons
        inner_opt = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            ang = params[3*i+2]
            inner_opt.append([x, y, ang])
        
        # Compute centroid
        cx = np.mean([p[0] for p in inner_opt])
        cy = np.mean([p[1] for p in inner_opt])
        
        # Find minimum S such that all fit in hexagon centered at (cx, cy) with angle 0
        # Binary search on S
        S_low = 1.5
        S_high = 10.0
        
        # Check feasibility
        for _ in range(30):
            S_mid = (S_low + S_high) / 2
            # Check all vertices
            feasible = True
            for (ix, iy, iang) in inner_opt:
                verts = hex_vertices(ix, iy, 1.0, iang)
                for (vx, vy) in verts:
                    if not point_in_hexagon(vx, vy, cx, cy, S_mid, 0.0):
                        feasible = False
                        break
                if not feasible:
                    break
            if feasible:
                S_high = S_mid
            else:
                S_low = S_mid
        
        return S_high
    
    # Set bounds for optimization
    bounds = []
    for p in inner:
        bounds.append((p[0] - 5, p[0] + 5))
        bounds.append((p[1] - 5, p[1] + 5))
        bounds.append((0, 360))
    
    # Use differential evolution for robustness
    result = differential_evolution(
        objective, bounds, seed=42, maxiter=50, tol=1e-6,
        popsize=15, mutation=(0.3, 1.2), recombination=0.9
    )
    
    if result.success or result.fun < 10:
        params = result.x
        inner_opt = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            ang = params[3*i+2]
            inner_opt.append([x, y, ang])
        
        cx = np.mean([p[0] for p in inner_opt])
        cy = np.mean([p[1] for p in inner_opt])
        
        # Find tight S
        S_low = 1.5
        S_high = 10.0
        for _ in range(50):
            S_mid = (S_low + S_high) / 2
            feasible = True
            for (ix, iy, iang) in inner_opt:
                verts = hex_vertices(ix, iy, 1.0, iang)
                for (vx, vy) in verts:
                    if not point_in_hexagon(vx, vy, cx, cy, S_mid, 0.0):
                        feasible = False
                        break
                if not feasible:
                    break
            if feasible:
                S_high = S_mid
            else:
                S_low = S_mid
        
        S_final = S_high
        return inner_opt, (cx, cy), S_final, 0.0
    
    # Fallback: return initial arrangement
    S_fallback = S_est + 0.5
    return inner, (0.0, 0.0), S_fallback, 0.0

