import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg):
    a = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300], dtype=float) * np.pi / 180.0 + a
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _max_hex_violation(verts, ox, oy, s, outer_angle_deg):
    """Max violation of hex containment. Returns 0 if all inside."""
    a = np.radians(outer_angle_deg)
    edge_dist = s * np.sqrt(3.0) / 2.0
    # 6 edge normals
    max_v = -1e9
    for k in range(6):
        theta = a + np.radians(30.0 + 60.0 * k)
        nx, ny = np.cos(theta), np.sin(theta)
        for vx, vy in verts:
            d = (vx - ox) * nx + (vy - oy) * ny - edge_dist
            if d > max_v:
                max_v = d
    return max_v

def construct():
    t0 = time.time()
    deadline = t0 + 135.0
    
    sqrt3 = np.sqrt(3.0)
    
    # 3x4 hex lattice, centered
    init = []
    for row in range(3):
        y = row * sqrt3
        if row % 2 == 0:
            xs = [-3.0, -1.0, 1.0, 3.0]
        else:
            xs = [-2.0, 0.0, 2.0, 4.0]
        for x in xs:
            init.append([x, y, 0.0])
    
    # Center
    mx = np.mean([p[0] for p in init])
    my = np.mean([p[1] for p in init])
    init = [[p[0]-mx, p[1]-my, 0.0] for p in init]
    
    best_s = 5.0
    best_inner = [list(p) for p in init]
    best_ox, best_oy = 0.0, 0.0
    best_oa = 0.0
    
    def obj(params):
        if time.time() > deadline:
            return 1e9
        inner_p = params[:36].reshape(12, 3)
        ox, oy, s, oa = params[36], params[37], params[38], params[39]
        if s < 1.0:
            return 1e6
        
        # Containment
        max_v = -1e9
        a = np.radians(oa)
        edge_dist = s * sqrt3 / 2.0
        normals = []
        for k in range(6):
            theta = a + np.radians(30.0 + 60.0 * k)
            normals.append((np.cos(theta), np.sin(theta)))
        
        for i in range(12):
            cx, cy, ang = inner_p[i]
            va = np.radians(ang)
            base_ang = np.array([0, 60, 120, 180, 240, 300], dtype=float) * np.pi / 180.0 + va
            vx = cx + np.cos(base_ang)
            vy = cy + np.sin(base_ang)
            for j in range(6):
                nx, ny = normals[j]
                d = (vx[j] - ox) * nx + (vy[j] - oy) * ny - edge_dist
                if d > max_v:
                    max_v = d
        
        # Overlap: min pairwise distance >= 2
        min_d = 1e9
        for i in range(12):
            for j in range(i+1, 12):
                dx = inner_p[i,0] - inner_p[j,0]
                dy = inner_p[i,1] - inner_p[j,1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < min_d:
                    min_d = d
        overlap_v = 2.0 - min_d
        
        total_v = max(max_v, overlap_v)
        if total_v > 0:
            return s + 10.0 * total_v
        return s
    
    x0 = np.zeros(40)
    for i in range(12):
        x0[i*3] = init[i][0]
        x0[i*3+1] = init[i][1]
        x0[i*3+2] = 0.0
    x0[36] = 0.0
    x0[37] = 0.0
    x0[38] = 5.0
    x0[39] = 0.0
    
    np.random.seed(42)
    for trial in range(3):
        if time.time() > deadline - 10.0:
            break
        x_trial = x0.copy()
        if trial > 0:
            x_trial[:36] += np.random.normal(0, 0.05, 36)
            x_trial[38] = 4.8 + 0.2 * trial
        try:
            res = minimize(obj, x_trial, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
            if res.fun < best_s:
                best_s = res.fun
                inner_p = res.x[:36].reshape(12, 3)
                best_inner = [list(inner_p[i]) for i in range(12)]
                best_ox, best_oy, best_oa = res.x[36], res.x[37], res.x[39]
        except Exception:
            pass
    
    return best_inner, (best_ox, best_oy), best_s, best_oa
# EVOLVE-BLOCK-END
