import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return [(cx + side * np.cos(a), cy + side * np.sin(a)) for a in angles]

def max_violation(px, py, cx, cy, side, angle_deg):
    """Return how far point is outside hexagon (0 if inside)."""
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(-angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    inradius = side * np.sqrt(3) / 2
    max_proj = 0
    for k in range(6):
        normal_angle = np.radians(30 + k * 60)
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        proj = lx * nx + ly * ny
        if proj > max_proj:
            max_proj = proj
    return max(0, max_proj - inradius)

def construct():
    # Place 11 hexagons in a compact arrangement
    inner = []
    inner.append([0.0, 0.0, 0.0])
    
    dist = np.sqrt(3)
    for k in range(6):
        angle = k * 60
        x = dist * np.cos(np.radians(angle))
        y = dist * np.sin(np.radians(angle))
        inner.append([x, y, 0.0])
    
    for k in range(4):
        angle = k * 60
        x = 2 * dist * np.cos(np.radians(angle))
        y = 2 * dist * np.sin(np.radians(angle))
        inner.append([x, y, 0.0])
    
    all_vertices = []
    for (ix, iy, iangle) in inner:
        verts = hexagon_vertices(ix, iy, 1.0, iangle)
        all_vertices.extend(verts)
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for outer_angle in [0, 15, 30, 45, 60]:
        centroid_x = np.mean([v[0] for v in all_vertices])
        centroid_y = np.mean([v[1] for v in all_vertices])
        
        def objective(params):
            cx, cy, side = params
            if side < 0:
                return 1e10
            max_viol = 0
            for (px, py) in all_vertices:
                viol = max_violation(px, py, cx, cy, side, outer_angle)
                if viol > max_viol:
                    max_viol = viol
            return side + 100 * max_viol
        
        x0 = [centroid_x, centroid_y, 5.0]
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success:
            cx, cy, side = res.x
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = outer_angle
    
    return inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
