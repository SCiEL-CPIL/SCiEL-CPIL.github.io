import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    vertices = np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])
    return vertices

def min_distance_between_hexagons(p1, a1, p2, a2, s1=1.0, s2=1.0):
    """
    Approximate minimum distance between two regular hexagons.
    For same orientation, min center distance for non-overlap is 2 * apothem = s*sqrt(3).
    We use a conservative estimate based on circumradius sum as a safe lower bound for separation,
    but for packing we want the actual minimum distance.
    
    For two regular hexagons of side s with relative angle delta, the minimum center distance
    to avoid overlap is approximately s * sqrt(3) when aligned. We'll use a simpler model:
    treat each hexagon as a circle of radius = apothem = s*sqrt(3)/2 for separation check,
    but this is conservative. Better: use the fact that for aligned hexagons, min dist = s*sqrt(3).
    """
    # For aligned hexagons, min center distance = 2 * apothem = s * sqrt(3)
    # For simplicity in optimization, use this as the constraint distance
    return s1 * np.sqrt(3)

def containment_constraint(cx, cy, angle_deg, side):
    """
    Check if a unit hexagon at (cx,cy) with angle angle_deg is fully inside
    the outer hexagon centered at (0,0) with given side and angle.
    Returns the maximum distance from origin to any vertex of inner hexagon,
    compared to the outer hexagon's inradius at that direction.
    """
    # Get inner hexagon vertices
    inner_verts = hexagon_vertices(cx, cy, 1.0, angle_deg)
    
    # For each vertex, check if it's inside the outer hexagon
    # Outer hexagon: center (0,0), side length, rotation
    outer_verts = hexagon_vertices(0, 0, side, angle_deg)
    
    # Point-in-polygon test for convex polygon
    def point_in_convex_poly(pt, poly):
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i+1) % n]
            # Cross product to check same side
            cross = (b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0])
            if cross < -1e-10:
                return False
        return True
    
    for v in inner_verts:
        if not point_in_convex_poly(v, outer_verts):
            return False
    return True

def objective(params):
    """
    params: [cx1, cy1, a1, cx2, cy2, a2, ..., cN, yN, aN, ox, oy, side, oangle]
    Total: 3*N + 4 = 3*11 + 4 = 37
    """
    # Deadline guard
    if not hasattr(objective, '_start_time'):
        objective._start_time = time.time()
    if time.time() - objective._start_time > 140:
        return 1e10  # Penalty to stop optimization
    
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append((x, y, a))
    
    ox = params[3*N]
    oy = params[3*N+1]
    side = params[3*N+2]
    oangle = params[3*N+3]
    
    # Shift so outer hexagon is at origin for simplicity
    # Translate all inner by -ox, -oy
    inner_shifted = [(x - ox, y - oy, a) for (x, y, a) in inner]
    
    # Penalty for non-overlap
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ai = inner_shifted[i]
            xj, yj, aj = inner_shifted[j]
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = np.sqrt(3)  # For aligned hexagons
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    # Penalty for containment: each inner hex must be inside outer
    # Use vertex check with penalty
    for i in range(N):
        xi, yi, ai = inner_shifted[i]
        inner_verts = hexagon_vertices(xi, yi, 1.0, ai)
        outer_verts = hexagon_vertices(0, 0, side, oangle)
        
        def point_in_convex_poly(pt, poly):
            n = len(poly)
            for k in range(n):
                a = poly[k]
                b = poly[(k+1) % n]
                cross = (b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0])
                if cross < -1e-10:
                    return False
            return True
        
        for v in inner_verts:
            if not point_in_convex_poly(v, outer_verts):
                # Distance to boundary approximation: use distance to closest edge
                # Simplified: penalize based on how far outside
                min_dist_to_edge = 1e10
                for k in range(6):
                    a = outer_verts[k]
                    b = outer_verts[(k+1) % 6]
                    # Distance from point v to line segment ab
                    ax, ay = a
                    bx, by = b
                    vx, vy = v
                    dx = bx - ax
                    dy = by - ay
                    len2 = dx*dx + dy*dy
                    if len2 > 0:
                        t = np.clip(((vx-ax)*dx + (vy-ay)*dy) / len2, 0, 1)
                        px = ax + t*dx
                        py = ay + t*dy
                        dist = np.sqrt((vx-px)**2 + (vy-py)**2)
                        min_dist_to_edge = min(min_dist_to_edge, dist)
                # Check if point is outside
                if not point_in_convex_poly(v, outer_verts):
                    penalty += (min_dist_to_edge + 0.1) ** 2
    
    return side + 100.0 * penalty

def construct():
    """Construct the packing of N unit hexagons inside a regular hexagon."""
    # Initial honeycomb arrangement
    inner = []
    # Center
    inner.append((0.0, 0.0, 0.0))
    # First ring: 6 hexagons at distance sqrt(3)
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        inner.append((x, y, 0.0))
    # Second ring: 4 more hexagons at distance 2*sqrt(3)
    for k in range(4):
        angle = np.radians(60 * (2*k + 1))  # Offset positions
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        inner.append((x, y, 0.0))
    
    # Initial outer hexagon: large enough
    side = 6.0
    center = (0.0, 0.0)
    oangle = 0.0
    
    # Build initial params
    params = []
    for (x, y, a) in inner:
        params.extend([x, y, a])
    params.extend([0.0, 0.0, side, oangle])
    params = np.array(params)
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))  # x
        bounds.append((-5, 5))  # y
        bounds.append((-180, 180))  # angle
    bounds.append((-5, 5))  # ox
    bounds.append((-5, 5))  # oy
    bounds.append((2.0, 10.0))  # side
    bounds.append((-180, 180))  # oangle
    
    # Optimize
    start_time = time.time()
    objective._start_time = start_time
    
    result = minimize(objective, params, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    if time.time() - start_time > 140:
        # Return initial guess if timeout
        return inner, center, side, oangle
    
    best_params = result.x
    best_side = best_params[3*N+2]
    
    # Extract best inner
    best_inner = []
    for i in range(N):
        x = best_params[3*i] - best_params[3*N]
        y = best_params[3*i+1] - best_params[3*N+1]
        a = best_params[3*i+2]
        best_inner.append([x, y, a])
    
    best_center = (0.0, 0.0)
    best_oangle = best_params[3*N+3]
    
    return best_inner, best_center, best_side, best_oangle
