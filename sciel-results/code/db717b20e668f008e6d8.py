import numpy as np
from scipy.optimize import minimize

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def min_enclosing_hexagon_side(inner_hexes, outer_angle_deg):
    """
    Given inner hexagons and a fixed outer hexagon orientation,
    find the minimum side length of a regular hexagon with that orientation
    that contains all inner hexagons.
    Returns (center_x, center_y, side)
    """
    outer_angle_rad = np.radians(outer_angle_deg)
    # 6 outward normals of the outer hexagon
    normals = []
    for k in range(6):
        ang = outer_angle_rad + k * np.pi / 3
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    
    # For each normal, compute min and max projection of all inner hexagon vertices
    max_projs = []
    min_projs = []
    for n in normals:
        max_p = -np.inf
        min_p = np.inf
        for (cx, cy, ang_deg, side) in inner_hexes:
            verts = hexagon_vertices(cx, cy, side, ang_deg)
            projs = verts @ n
            max_p = max(max_p, np.max(projs))
            min_p = min(min_p, np.min(projs))
        max_projs.append(max_p)
        min_projs.append(min_p)
    
    # The outer hexagon with center (ox, oy) and side s has half-width s*sqrt(3)/2 in each normal direction
    # We need: min_p[k] >= ox*n_kx + oy*n_ky - s*sqrt(3)/2
    #          max_p[k] <= ox*n_kx + oy*n_ky + s*sqrt(3)/2
    
    # To minimize s, we should center the outer hexagon appropriately.
    # Let's use the centroid of all inner hexagon centers as a starting point
    centers = np.array([(cx, cy) for (cx, cy, ang_deg, side) in inner_hexes])
    ox, oy = np.mean(centers, axis=0)
    
    # Compute required side
    s_required = 0.0
    for k in range(6):
        n = normals[k]
        proj_center = ox * n[0] + oy * n[1]
        # Required half-width in this direction
        half_w1 = max_projs[k] - proj_center
        half_w2 = proj_center - min_projs[k]
        half_w = max(half_w1, half_w2)
        # side = half_w / (sqrt(3)/2)
        s_k = half_w / (np.sqrt(3) / 2)
        s_required = max(s_required, s_k)
    
    return ox, oy, s_required

def min_enclosing_hexagon(inner_hexes):
    """Find minimum enclosing regular hexagon over all orientations."""
    best_side = np.inf
    best_params = None
    
    for outer_angle_deg in np.linspace(0, 60, 13):
        ox, oy, side = min_enclosing_hexagon_side(inner_hexes, outer_angle_deg)
        if side < best_side:
            best_side = side
            best_params = (ox, oy, side, outer_angle_deg)
    
    return best_params

def construct():
    sqrt3 = np.sqrt(3)
    
    # Compact 11-hexagon arrangement: center + 6 ring + 4 outer
    base_positions = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, sqrt3/2),
        (-0.5, sqrt3/2),
        (-1.0, 0.0),
        (-0.5, -sqrt3/2),
        (0.5, -sqrt3/2),
        (2.0, 0.0),
        (1.0, sqrt3),
        (-1.0, sqrt3),
        (-2.0, 0.0),
    ]
    
    def objective(params):
        inner_hexes_opt = []
        for i in range(11):
            dx = params[3*i]
            dy = params[3*i+1]
            dangle = params[3*i+2]
            x = base_positions[i][0] + dx
            y = base_positions[i][1] + dy
            inner_hexes_opt.append((x, y, dangle, 1.0))
        
        # Overlap check
        for i in range(11):
            for j in range(i+1, 11):
                xi, yi, ai, _ = inner_hexes_opt[i]
                xj, yj, aj, _ = inner_hexes_opt[j]
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if dist < 1.7:
                    return 100.0 + (1.7 - dist) * 10
        
        _, _, side, _ = min_enclosing_hexagon(inner_hexes_opt)
        return side
    
    x0 = np.zeros(33)
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    best_params = result.x
    inner_hexes_final = []
    for i in range(11):
        dx = best_params[3*i]
        dy = best_params[3*i+1]
        dangle = best_params[3*i+2]
        x = base_positions[i][0] + dx
        y = base_positions[i][1] + dy
        inner_hexes_final.append((x, y, dangle, 1.0))
    
    ox, oy, side, outer_angle = min_enclosing_hexagon(inner_hexes_final)
    
    inner = [[x, y, ang_deg] for (x, y, ang_deg, s) in inner_hexes_final]
    return inner, (ox, oy), side, outer_angle
