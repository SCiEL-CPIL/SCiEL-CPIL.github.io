import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def point_in_hex(px, py, cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    for i in range(6):
        a = angle + i * np.pi / 3
        dx = px - cx
        dy = py - cy
        proj = dx * np.cos(a) + dy * np.sin(a)
        if proj > side * np.cos(np.pi / 6):
            return False
    return True

def hex_contains_hex(hx, hy, ha_deg, S, cx, cy, ca_deg):
    inner_verts = hex_vertices(hx, hy, 1.0, ha_deg)
    for vx, vy in inner_verts:
        if not point_in_hex(vx, vy, cx, cy, S, ca_deg):
            return False
    return True

def hex_overlaps(h1, h2):
    x1, y1, a1 = h1
    x2, y2, a2 = h2
    dx = x1 - x2
    dy = y1 - y2
    dist = np.sqrt(dx * dx + dy * dy)
    if dist > 2.0:
        return False
    if dist < np.sqrt(3) - 0.001:
        return True
    v1 = hex_vertices(x1, y1, 1.0, a1)
    v2 = hex_vertices(x2, y2, 1.0, a2)
    for vx, vy in v1:
        if point_in_hex(vx, vy, x2, y2, 1.0, a2):
            return True
    for vx, vy in v2:
        if point_in_hex(vx, vy, x1, y1, 1.0, a1):
            return True
    return False

def check_packing(inner, cx, cy, S, ca_deg):
    for hx, hy, ha in inner:
        if not hex_contains_hex(hx, hy, ha, S, cx, cy, ca_deg):
            return False
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            if hex_overlaps(inner[i], inner[j]):
                return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice arrangement
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    inner = []
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
    
    cx0 = np.mean([p[0] for p in inner])
    cy0 = np.mean([p[1] for p in inner])
    inner_c = [[p[0] - cx0, p[1] - cy0, p[2]] for p in inner]
    
    best_S = 40.0
    best_params = [0.0, 0.0, 3.5, 0.0]
    
    # Grid search over rotation and center offsets
    for ca_init in [0, 30, 60, 90, 15, 45, 75]:
        for cx_init in np.arange(-0.8, 0.9, 0.2):
            for cy_init in np.arange(-0.8, 0.9, 0.2):
                S_lo, S_hi = 2.0, 6.0
                for _ in range(25):
                    S_mid = (S_lo + S_hi) / 2
                    if check_packing(inner_c, cx_init, cy_init, S_mid, ca_init):
                        S_hi = S_mid
                    else:
                        S_lo = S_mid
                if S_hi < best_S:
                    best_S = S_hi
                    best_params = [cx_init, cy_init, S_hi, ca_init]
    
    # Local refinement
    if time.time() < deadline:
        def obj_refine(params):
            cx, cy, S, ca_deg = params
            if S < 1.5:
                return 100.0
            if not check_packing(inner_c, cx, cy, S, ca_deg):
                return 100.0 + S
            return S
        
        res = minimize(obj_refine, best_params, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9})
        if res.fun < best_S:
            best_params = res.x
            best_S = res.fun
    
    # Differential evolution if time permits
    if time.time() < deadline - 40:
        def obj_de(x):
            cx, cy, S, ca_deg = x
            if S < 1.5:
                return 100.0
            if not check_packing(inner_c, cx, cy, S, ca_deg):
                return 100.0 + S
            return S
        
        bounds = [(-2, 2), (-2, 2), (2.8, 4.5), (0, 180)]
        de_result = differential_evolution(
            obj_de, bounds, maxiter=30, popsize=12,
            tol=1e-4, seed=42, polish=True
        )
        if de_result.fun < best_S:
            best_params = de_result.x
            best_S = de_result.fun
    
    cx, cy, S, ca_deg = best_params
    outer_center = (cx0 + cx, cy0 + cy)
    
    return inner, outer_center, S, ca_deg
# EVOLVE-BLOCK-END
