import numpy as np
from scipy.optimize import minimize
import time

N = 12

def get_vertices(cx, cy, theta):
    angles = theta + np.arange(6) * np.pi / 3
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def calc_overlap_penalty(positions):
    n = len(positions)
    penalty = 0.0
    verts = [get_vertices(positions[i, 0], positions[i, 1], positions[i, 2]) for i in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = verts[i]
            vj = verts[j]
            
            theta_i = positions[i, 2]
            normals_i = theta_i + np.pi/6 + np.arange(6) * np.pi/3
            ni = np.column_stack([np.cos(normals_i), np.sin(normals_i)])
            
            theta_j = positions[j, 2]
            normals_j = theta_j + np.pi/6 + np.arange(6) * np.pi/3
            nj = np.column_stack([np.cos(normals_j), np.sin(normals_j)])
            
            all_normals = np.vstack([ni, nj])
            
            proj_i = np.dot(vi, all_normals.T)
            proj_j = np.dot(vj, all_normals.T)
            
            min_i = np.min(proj_i, axis=1)
            max_i = np.max(proj_i, axis=1)
            min_j = np.min(proj_j, axis=1)
            max_j = np.max(proj_j, axis=1)
            
            overlap = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
            min_overlap = np.min(overlap)
            
            if min_overlap > 0:
                penalty += min_overlap ** 2
                
    return penalty

def objective(vars):
    positions = vars.reshape(N, 3)
    
    verts_list = [get_vertices(positions[i, 0], positions[i, 1], positions[i, 2]) for i in range(N)]
    all_verts = np.vstack(verts_list)
    
    outer_angles = np.pi/6 + np.arange(6) * np.pi/3
    n_outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    
    proj = np.dot(all_verts, n_outer.T)
    max_proj = np.max(np.abs(proj))
    S = 2.0 / np.sqrt(3.0) * max_proj
    
    pen = calc_overlap_penalty(positions)
    
    return S + 50.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = float('inf')
    best_vars = None
    
    e1 = np.array([np.sqrt(3), 0])
    e2 = np.array([np.sqrt(3)/2, 1.5])
    
    points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * e1 + j * e2
            points.append(p)
    points = np.array(points)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:N]
    init_positions = points[idx]
    
    init_vars = np.zeros((N, 3))
    init_vars[:, :2] = init_positions
    init_vars[:, 2] = 0.0
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        if restarts == 1:
            current_vars = init_vars.flatten()
        else:
            current_vars = init_vars.flatten() + np.random.normal(0, 0.5, size=N*3)
            current_vars[2::3] = np.random.uniform(0, 2*np.pi, size=N)
            
        try:
            res = minimize(objective, current_vars, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            final_vars = res.x
            positions = final_vars.reshape(N, 3)
            pen = calc_overlap_penalty(positions)
            if pen < 1e-3:
                verts_list = [get_vertices(positions[i, 0], positions[i, 1], positions[i, 2]) for i in range(N)]
                all_verts = np.vstack(verts_list)
                outer_angles = np.pi/6 + np.arange(6) * np.pi/3
                n_outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
                proj = np.dot(all_verts, n_outer.T)
                max_proj = np.max(np.abs(proj))
                S = 2.0 / np.sqrt(3.0) * max_proj
                
                if S < best_score:
                    best_score = S
                    best_vars = final_vars
        except Exception:
            pass

        if restarts > 20:
            break

    if best_vars is None:
        best_vars = init_vars.flatten()
        positions = best_vars.reshape(N, 3)
        verts_list = [get_vertices(positions[i, 0], positions[i, 1], positions[i, 2]) for i in range(N)]
        all_verts = np.vstack(verts_list)
        outer_angles = np.pi/6 + np.arange(6) * np.pi/3
        n_outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        proj = np.dot(all_verts, n_outer.T)
        max_proj = np.max(np.abs(proj))
        best_score = 2.0 / np.sqrt(3.0) * max_proj

    inner = []
    for i in range(N):
        x = best_vars[3*i]
        y = best_vars[3*i+1]
        angle = np.degrees(best_vars[3*i+2]) % 360.0
        inner.append([x, y, angle])
    
    return inner, (0.0, 0.0), best_score, 0.0
