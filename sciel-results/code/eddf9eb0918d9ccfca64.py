"""Packing 12 unit hexagons in a minimal outer hexagon using hexagonal lattice + optimization."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import math

N = 12

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side length, and rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * math.pi / 3
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side, rotation."""
    # Transform to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle = math.radians(angle_deg)
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side s, the condition for a point to be inside:
    # The hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
    # In local coords, the hexagon edges are at distances s*cos(30) = s*sqrt(3)/2 from center
    # along the 6 edge normals (at 30, 90, 150, 210, 270, 330 degrees)
    
    inradius = side * math.sqrt(3) / 2
    
    for i in range(6):
        edge_angle = math.radians(30 + i * 60)
        # Distance from point to the edge line (signed)
        dist = lx * math.cos(edge_angle) + ly * math.sin(edge_angle)
        if dist > inradius:
            return False
    return True

def _hexagon_inside_hexagon(inner_center, inner_angle, outer_center, outer_side, outer_angle):
    """Check if a unit hexagon is fully inside the outer hexagon."""
    ix, iy = inner_center
    ox, oy = outer_center
    # Check all 6 vertices of inner hexagon are inside outer hexagon
    iverts = _hexagon_vertices(ix, iy, 1.0, inner_angle)
    for vx, vy in iverts:
        if not _point_in_hexagon(vx, vy, ox, oy, outer_side, outer_angle):
            return False
    return True

def _objective(params):
    """Objective function for optimization: minimize outer hexagon side length."""
    n = 12
    inner = []
    for i in range(n):
        ix = params[3*i]
        iy = params[3*i+1]
        iangle = params[3*i+2]
        inner.append((ix, iy, iangle))
    
    ox = params[3*n]
    oy = params[3*n+1]
    oangle = params[3*n+2]
    oside = params[3*n+3]
    
    if oside < 0.5:
        return 1e10
    
    for ix, iy, iangle in inner:
        if not _hexagon_inside_hexagon((ix, iy), iangle, (ox, oy), oside, oangle):
            return 1e10
    
    return oside

def construct():
    """Construct 12 unit hexagons in a hexagonal lattice arrangement, then optimize."""
    
    # Hexagonal lattice arrangement for 12 hexagons
    # Pointy-top hexagons: width = sqrt(3)*side, height = 2*side
    side = 1.0
    h_spacing = math.sqrt(3) * side
    v_spacing = 1.5 * side
    
    positions = []
    # 3 rows of 4 hexagons
    for row in range(3):
        y = row * v_spacing
        offset = (h_spacing / 2) if (row % 2 == 1) else 0
        for col in range(4):
            x = col * h_spacing + offset
            positions.append([x, y, 0.0])
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    # Bounds for optimization
    bounds = []
    for i in range(12):
        bounds.append((-4, 4))
        bounds.append((-4, 4))
        bounds.append((0, 60))
    bounds.append((-4, 4))
    bounds.append((-4, 4))
    bounds.append((0, 60))
    bounds.append((1.5, 8.0))
    
    x0 = []
    for p in positions:
        x0.extend([p[0], p[1], p[2]])
    x0.extend([0.0, 0.0, 0.0, 4.0])
    
    # Use Nelder-Mead for speed
    best_fun = 1e10
    best_x = x0.copy()
    
    # Try multiple initial outer angles
    for oangle_init in [0, 30, 60]:
        x0_trial = x0.copy()
        x0_trial[38] = oangle_init
        
        res = minimize(_objective, x0_trial, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x.copy()
    
    if best_fun < 1e9:
        inner_opt = []
        for i in range(12):
            inner_opt.append([best_x[3*i], best_x[3*i+1], best_x[3*i+2]])
        outer_x = best_x[36]
        outer_y = best_x[37]
        outer_angle = best_x[38]
        outer_side = best_x[39]
    else:
        # Fallback
        inner_opt = positions
        outer_x, outer_y, outer_angle = 0.0, 0.0, 0.0
        # Estimate outer side from bounding box
        max_x = max(p[0] for p in positions) + 1.0
        max_y = max(p[1] for p in positions) + 1.0
        outer_side = max(max_x, max_y) * 1.5
    
    return inner_opt, (outer_x, outer_y), outer_side, outer_angle
# EVOLVE-BLOCK-END
