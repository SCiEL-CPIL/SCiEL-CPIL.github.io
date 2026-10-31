import numpy as np
from scipy.optimize import minimize
import time

N = 11

def get_hex_vertices(center, angle_deg, side=1.0):
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) + theta
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def min_container_side(vertices):
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    d = vertices @ normals.T
    max_d = np.max(np.abs(d))
    S = 2.0 / np.sqrt(3.0) * max_d
    return S

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    coords = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (1,-2), (-2,1), (-1,-1)]
    positions = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        positions.append([x, y])
    
    angles = [0.0] * N
    x0 = np.array(positions).flatten()
    a0 = np.array(angles)
    p0 = np.concatenate((x0, a0))
    
    min_dist = np.sqrt(3.0)
    
    def objective(p):
        pos = p[:2*N].reshape(N, 2)
        ang = p[2*N:]
        
        centroid = np.mean(pos, axis=0)
        pos_centered = pos - centroid
        
        verts_list = []
        for i in range(N):
            v = get_hex_vertices(pos_centered[i], ang[i])
            verts_list.append(v)
        all_verts = np.vstack(verts_list)
        S = min_container_side(all_verts)
        
        # Check disjointness
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                diff = pos_centered[i] - pos_centered[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
                    # If they overlap significantly, return large value
                    if dist < min_dist * 0.5:
                        return 1000.0 + penalty
        
        return S + penalty * 10.0

    if time.time() - start_time > deadline - safety_margin:
        S_init = objective(p0)
        return [[pos[0], pos[1], 0.0] for pos in positions], (0.0, 0.0), S_init, 0.0

    try:
        res = minimize(objective, p0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-6})
        
        p_opt = res.x
        pos_opt = p_opt[:2*N].reshape(N, 2)
        ang_opt = p_opt[2*N:]
        
        centroid = np.mean(pos_opt, axis=0)
        pos_final = pos_opt - centroid
        
        # Verify validity
        final_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                diff = pos_final[i] - pos_final[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist:
                    final_penalty += (min_dist - dist) ** 2
        
        if final_penalty > 1e-4:
            # Fallback to initial
            S_init = objective(p0)
            return [[pos[0], pos[1], 0.0] for pos in positions], (0.0, 0.0), S_init, 0.0
            
        S_final = objective(p_opt)
        
        inner = []
        for i in range(N):
            inner.append([pos_final[i][0], pos_final[i][1], ang_opt[i]])
            
        return inner, (0.0, 0.0), S_final, 0.0

    except Exception:
        S_init = objective(p0)
        return [[pos[0], pos[1], 0.0] for pos in positions], (0.0, 0.0), S_init, 0.0
