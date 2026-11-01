"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        theta = i * np.pi / 3
        nx = np.cos(theta)
        ny = np.sin(theta)
        dist = lx * nx + ly * ny
        if dist > apothem + 1e-10:
            return False
    return True

def hexagon_contains_hexagon(inner_center, inner_angle, outer_center, outer_side, outer_angle):
    inner_verts = hexagon_vertices(inner_center[0], inner_center[1], 1.0, inner_angle)
    for v in inner_verts:
        if not point_in_hexagon(v[0], v[1], outer_center[0], outer_center[1], outer_side, outer_angle):
            return False
    return True

def check_all_fit(inner_positions, inner_angles, outer_center, outer_side, outer_angle):
    for i in range(len(inner_positions)):
        cx, cy = inner_positions[i]
        angle = inner_angles[i]
        if not hexagon_contains_hexagon([cx, cy], angle, outer_center, outer_side, outer_angle):
            return False
    return True

def objective(params):
    n = N
    outer_cx = params[0]
    outer_cy = params[1]
    outer_side = params[2]
    outer_angle = params[3]
    
    penalty = 0.0
    for i in range(n):
        base = 4 + i * 3
        cx = params[base]
        cy = params[base + 1]
        angle = params[base + 2]
        inner_verts = hexagon_vertices(cx, cy, 1.0, angle)
        for v in inner_verts:
            ang = np.radians(outer_angle)
            dx = v[0] - outer_cx
            dy = v[1] - outer_cy
            cos_a = np.cos(-ang)
            sin_a = np.sin(-ang)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            apothem = outer_side * np.sqrt(3) / 2
            for j in range(6):
                theta = j * np.pi / 3
                nx = np.cos(theta)
                ny = np.sin(theta)
                dist = lx * nx + ly * ny
                if dist > apothem:
                    penalty += (dist - apothem) ** 2
    
    return outer_side + 100.0 * penalty

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    init_positions = []
    init_angles = []
    
    for x in [-sqrt3, 0, sqrt3]:
        init_positions.append([x, 0.0])
        init_angles.append(0.0)
    
    for x in [-1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3]:
        init_positions.append([x, 1.5])
        init_angles.append(0.0)
    
    for x in [-1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3]:
        init_positions.append([x, -1.5])
        init_angles.append(0.0)
    
    outer_cx_init = 0.0
    outer_cy_init = 0.0
    outer_side_init = 10.0
    outer_angle_init = 0.0
    
    params = [outer_cx_init, outer_cy_init, outer_side_init, outer_angle_init]
    for pos, ang in zip(init_positions, init_angles):
        params.append(pos[0])
        params.append(pos[1])
        params.append(ang)
    
    result = minimize(objective, params, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_params = result.x
    outer_cx = best_params[0]
    outer_cy = best_params[1]
    outer_side = best_params[2]
    outer_angle = best_params[3]
    
    inner = []
    for i in range(N):
        base = 4 + i * 3
        x = best_params[base]
        y = best_params[base + 1]
        a = best_params[base + 2]
        inner.append([x, y, a])
    
    return inner, (outer_cx, outer_cy), outer_side, outer_angle
# EVOLVE-BLOCK-END
