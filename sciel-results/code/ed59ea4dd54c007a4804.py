"""Rewritten program for packing 11 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import math

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        vertices.append((x, y))
    return vertices

def point_in_polygon(px, py, polygon):
    """Ray casting algorithm to check if point is inside polygon."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def segment_intersection(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def polygons_intersect(poly1, poly2):
    """Check if two polygons intersect (including one containing the other)."""
    # Check if any vertex of poly1 is inside poly2
    for v in poly1:
        if point_in_polygon(v[0], v[1], poly2):
            return True
    # Check if any vertex of poly2 is inside poly1
    for v in poly2:
        if point_in_polygon(v[0], v[1], poly1):
            return True
    # Check for edge intersections
    n1 = len(poly1)
    n2 = len(poly2)
    for i in range(n1):
        for j in range(n2):
            if segment_intersection(poly1[i], poly1[(i+1)%n1], poly2[j], poly2[(j+1)%n2]):
                return True
    return False

def outer_hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of outer regular hexagon."""
    return hexagon_vertices(cx, cy, side, angle_deg)

def check_inside(inner_hex, outer_poly):
    """Check if all vertices of inner hexagon are inside outer polygon."""
    for v in inner_hex:
        if not point_in_polygon(v[0], v[1], outer_poly):
            return False
    return True

def compute_objective(params, N):
    """
    Compute objective function: outer side length + penalties.
    params: [x1, y1, a1, ..., xN, yN, aN, L]
    Outer hexagon is fixed at center (0,0) and angle 0.
    """
    L = params[-1]
    if L <= 0:
        return 1e10
    
    # Outer hexagon is fixed at (0,0) with side L and angle 0
    outer_poly = outer_hexagon_vertices(0.0, 0.0, L, 0.0)
    
    penalty = 0.0
    inner_polys = []
    
    for i in range(N):
        ix = params[3*i]
        iy = params[3*i+1]
        iangle = params[3*i+2]
        poly = hexagon_vertices(ix, iy, 1.0, iangle)
        inner_polys.append(poly)
        
        # Check if inner hexagon is inside outer
        if not check_inside(poly, outer_poly):
            # Calculate max distance of vertices from center to estimate penalty
            # Or just use a fixed penalty
            penalty += 100.0
            # More precise penalty: how far out is it?
            # For simplicity, keep fixed penalty but large
            
    # Check pairwise non-intersection
    for i in range(N):
        for j in range(i+1, N):
            if polygons_intersect(inner_polys[i], inner_polys[j]):
                penalty += 100.0
                
    return L + penalty

def construct():
    """
    Construct 11 unit hexagons packed in a regular hexagon, minimizing outer side length.
    Returns: (inner, center, side, angle)
    """
    N_local = 11
    sqrt3 = math.sqrt(3)
    
    # Initial guess: 1-2-3-2-1 cluster + 2 defect hexagons
    # This creates an asymmetric shape that might pack tighter
    init = []
    
    # 1-2-3-2-1 cluster
    # Row 0 (y = 2*sqrt3): 1
    init.append((0.0, 2*sqrt3, 0.0))
    # Row 1 (y = sqrt3): 2
    init.append((-sqrt3/2, sqrt3, 0.0))
    init.append((sqrt3/2, sqrt3, 0.0))
    # Row 2 (y = 0): 3
    init.append((-sqrt3, 0.0, 0.0))
    init.append((0.0, 0.0, 0.0))
    init.append((sqrt3, 0.0, 0.0))
    # Row 3 (y = -sqrt3): 2
    init.append((-sqrt3/2, -sqrt3, 0.0))
    init.append((sqrt3/2, -sqrt3, 0.0))
    # Row 4 (y = -2*sqrt3): 1
    init.append((0.0, -2*sqrt3, 0.0))
    
    # 2 Defect hexagons
    # Place them in the gaps on the right and left, rotated 30 deg to fill voids
    # Gap location approx (1.5*sqrt3, 0.5*sqrt3) and (-1.5*sqrt3, 0.5*sqrt3)
    # Let's try placing them slightly inside the convex hull of the 9
    # The "corners" of the 1-2-3-2-1 are at (0, 2sqrt3+1) etc.
    # The side length of the bounding hexagon for 9 is approx 2*sqrt3 + 1 = 4.46
    # Let's place the 10th at (1.5*sqrt3, 0.5*sqrt3) with 30 deg rotation
    init.append((1.5*sqrt3, 0.5*sqrt3, 30.0))
    # 11th at (-1.5*sqrt3, 0.5*sqrt3) with 30 deg rotation
    init.append((-1.5*sqrt3, 0.5*sqrt3, 30.0))
    
    # Initial L estimate
    # The cluster extends to y=2*sqrt3+1 and x=1.5*sqrt3+1 (approx)
    # L should be at least max distance from center to any vertex
    # For now, guess 5.0
    L_init = 5.0
    
    x0 = np.zeros(3*N_local + 1)
    for i in range(N_local):
        x0[3*i] = init[i][0]
        x0[3*i+1] = init[i][1]
        x0[3*i+2] = init[i][2]
    x0[3*N_local] = L_init
    
    def objective(params):
        return compute_objective(params, N_local)
    
    best_params = x0.copy()
    best_val = objective(x0)
    
    # Multi-start optimization
    np.random.seed(42)
    
    for restart in range(10):
        if restart == 0:
            x0_cur = x0.copy()
        else:
            x0_cur = x0.copy()
            # Perturb positions
            noise = np.random.normal(0, 0.2, size=3*N_local)
            x0_cur[:3*N_local] += noise
            # Perturb L slightly
            x0_cur[3*N_local] = max(4.0, x0_cur[3*N_local] + np.random.normal(0, 0.1))
            
        try:
            result = minimize(
                objective, 
                x0_cur, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_params = result.x
        except Exception:
            pass
    
    # Extract best parameters
    inner = []
    for i in range(N_local):
        ix = best_params[3*i]
        iy = best_params[3*i+1]
        iangle = best_params[3*i+2]
        inner.append([ix, iy, iangle])
    
    outer_side_best = best_params[3*N_local]
    
    # Ensure outer side is positive
    outer_side_best = max(outer_side_best, 3.0)
    
    # Final validation
    outer_poly = outer_hexagon_vertices(0.0, 0.0, outer_side_best, 0.0)
    inner_polys = []
    for i in range(N_local):
        poly = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        inner_polys.append(poly)
        if not check_inside(poly, outer_poly):
            # If invalid, use a safe fallback
            return [[4.0 * k, 0.0, 0.0] for k in range(-(N_local // 2), N_local - N_local // 2)], (0.0, 0.0), 40.0, 0.0
    
    for i in range(N_local):
        for j in range(i+1, N_local):
            if polygons_intersect(inner_polys[i], inner_polys[j]):
                return [[4.0 * k, 0.0, 0.0] for k in range(-(N_local // 2), N_local - N_local // 2)], (0.0, 0.0), 40.0, 0.0
    
    return inner, (0.0, 0.0), outer_side_best, 0.0

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
