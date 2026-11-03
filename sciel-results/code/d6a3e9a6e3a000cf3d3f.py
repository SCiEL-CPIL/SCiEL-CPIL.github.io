import numpy as np
from scipy.optimize import minimize

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    theta = np.radians(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def construct():
    sqrt3 = np.sqrt(3)
    
    positions = [
        (0.0, 0.0),
        (sqrt3, 0.0),
        (-sqrt3, 0.0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (sqrt3/2, -1.5),
        (-sqrt3/2, -1.5),
        (3*sqrt3/2, 1.5),
        (3*sqrt3/2, -1.5),
        (-3*sqrt3/2, 1.5),
        (-3*sqrt3/2, -1.5),
    ]
    
    inner = [[float(p[0]), float(p[1]), 0.0] for p in positions]
    
    all_verts_list = []
    for ix, iy, ia in inner:
        verts = hex_vertices(ix, iy, 1.0, ia)
        all_verts_list.append(verts)
    all_verts = np.vstack(all_verts_list)
    
    def min_side_for_params(cx, cy, angle_deg):
        theta = np.radians(angle_deg)
        edge_angles = np.pi/6 + np.arange(6) * np.pi/3
        dx = all_verts[:, 0] - cx
        dy = all_verts[:, 1] - cy
        rx = dx * np.cos(-theta) - dy * np.sin(-theta)
        ry = dx * np.sin(-theta) + dy * np.cos(-theta)
        
        max_proj = 0.0
        for ea in edge_angles:
            proj = rx * np.cos(ea) + ry * np.sin(ea)
            mp = np.max(proj)
            if mp > max_proj:
                max_proj = mp
        return 2.0 * max_proj / sqrt3
    
    def objective(params):
        cx, cy, s, angle = params
        if s < 0.1:
            return 1e6
        s_needed = min_side_for_params(cx, cy, angle)
        violation = max(0.0, s_needed - s)
        return s + 100.0 * violation
    
    best_obj = float('inf')
    best_params = [0.0, 0.0, 5.0, 0.0]
    
    for s0 in [3.0, 4.0, 5.0, 6.0]:
        for a0 in [0.0, 30.0, 60.0, 90.0]:
            x0 = [0.0, 0.0, s0, a0]
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_params = res.x
            except:
                pass
    
    cx, cy, s, angle = best_params
    if s < 0.1:
        s = 5.0
    
    return inner, (float(cx), float(cy)), float(s), float(angle)
