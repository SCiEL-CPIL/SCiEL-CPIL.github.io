"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def max_violation(pos_arr, ang_arr, ox, oy, S, oang_deg):
    angle = np.radians(oang_deg)
    max_v = 0
    for i in range(len(pos_arr)):
        cx, cy = pos_arr[i]
        ang = ang_arr[i]
        verts = hex_vertices(cx, cy, 1.0, ang)
        for v in verts:
            dx, dy = v[0] - ox, v[1] - oy
            for k in range(6):
                na = angle + k * np.pi / 3
                proj = dx * np.cos(na) + dy * np.sin(na)
                viol = proj - S * np.sqrt(3) / 2
                if viol > max_v:
                    max_v = viol
    return max_v

def min_outer_side(pos_arr, ang_arr, ox, oy, oang_deg):
    angle = np.radians(oang_deg)
    max_proj = 0
    for i in range(len(pos_arr)):
        cx, cy = pos_arr[i]
        ang = ang_arr[i]
        verts = hex_vertices(cx, cy, 1.0, ang)
        for v in verts:
            dx, dy = v[0] - ox, v[1] - oy
            for k in range(6):
                na = angle + k * np.pi / 3
                proj = dx * np.cos(na) + dy * np.sin(na)
                if proj > max_proj:
                    max_proj = proj
    return max_proj / (np.sqrt(3) / 2)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    best_score = float('inf')
    best_result = None
    
    clusters = [
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,0),(-2,0),(0,2),(0,-2),(2,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,-1),(1,-1),(2,-1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,-1),(1,-1),(2,-1),(3,-1)],
        [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(-1,0),(-1,1),(-1,2),(-1,3)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,0),(2,-1),(1,-2),(-1,-2),(-2,-1)],
    ]
    
    for cluster in clusters:
        if time.time() > deadline:
            break
        positions = []
        for q, r in cluster:
            x = np.sqrt(3) * (q + r / 2)
            y = 1.5 * r
            positions.append([x, y])
        positions = np.array(positions)
        inner_angles = [0.0] * N
        com = np.mean(positions, axis=0)
        
        for outer_angle in range(0, 180, 15):
            if time.time() > deadline:
                break
            S_est = min_outer_side(positions, inner_angles, com[0], com[1], outer_angle)
            
            def obj(params):
                ox, oy, S = params
                v = max_violation(positions, inner_angles, ox, oy, S, outer_angle)
                return S + 100 * max(0, v)
            
            res = minimize(obj, [com[0], com[1], S_est], method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < S_est:
                ox, oy, S = res.x
                v = max_violation(positions, inner_angles, ox, oy, S, outer_angle)
                if v <= 1e-9 and S < best_score:
                    best_score = S
                    best_result = (positions.copy(), inner_angles.copy(), (ox, oy), S, outer_angle)
    
    if time.time() < deadline - 30:
        if best_result is None:
            cluster = clusters[0]
            positions = []
            for q, r in cluster:
                x = np.sqrt(3) * (q + r / 2)
                y = 1.5 * r
                positions.append([x, y])
            positions = np.array(positions)
            inner_angles = [0.0] * N
            com = np.mean(positions, axis=0)
            outer_angle = 0.0
            S_est = min_outer_side(positions, inner_angles, com[0], com[1], outer_angle)
            best_score = S_est
            best_result = (positions.copy(), inner_angles.copy(), (com[0], com[1]), S_est, outer_angle)
        else:
            positions = best_result[0].copy()
            inner_angles = best_result[1].copy()
            com = best_result[2]
            S_est = best_result[3]
            outer_angle = best_result[4]
        
        def obj_wrap(params):
            pos_flat = params[:2*N]
            angles = params[2*N:2*N+N]
            ox, oy = params[2*N+N:2*N+N+2]
            S = params[2*N+N+2]
            oang = params[2*N+N+3]
            pos_arr = pos_flat.reshape(N, 2)
            v = max_violation(pos_arr, angles, ox, oy, S, oang)
            return S + 100 * max(0, v)
        
        init_params = np.concatenate([positions.flatten(), inner_angles, [com[0], com[1]], [S_est], [outer_angle]])
        res = minimize(obj_wrap, init_params, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9})
        
        if res.fun < best_score:
            pos_flat = res.x[:2*N]
            angles = res.x[2*N:2*N+N]
            ox, oy = res.x[2*N+N:2*N+N+2]
            S = res.x[2*N+N+2]
            oang = res.x[2*N+N+3]
            pos_arr = pos_flat.reshape(N, 2)
            v = max_violation(pos_arr, angles, ox, oy, S, oang)
            if v <= 1e-9:
                best_score = S
                best_result = (pos_arr.copy(), angles.copy(), (ox, oy), S, oang)
    
    if time.time() < deadline - 15:
        cluster = clusters[0]
        positions = []
        for q, r in cluster:
            x = np.sqrt(3) * (q + r / 2)
            y = 1.5 * r
            positions.append([x, y])
        positions = np.array(positions)
        inner_angles = [30.0] * N
        com = np.mean(positions, axis=0)
        
        for outer_angle in range(0, 180, 15):
            if time.time() > deadline:
                break
            S_est = min_outer_side(positions, inner_angles, com[0], com[1], outer_angle)
            v = max_violation(positions, inner_angles, com[0], com[1], S_est, outer_angle)
            if v <= 1e-9 and S_est < best_score:
                best_score = S_est
                best_result = (positions.copy(), inner_angles.copy(), (com[0], com[1]), S_est, outer_angle)
    
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    pos_arr, angles, (ox, oy), S, oang = best_result
    inner = []
    for i in range(N):
        inner.append([pos_arr[i][0], pos_arr[i][1], angles[i]])
    
    return inner, (ox, oy), S, oang
# EVOLVE-BLOCK-END
