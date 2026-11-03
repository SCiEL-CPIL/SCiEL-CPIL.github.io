import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0 # Leave 10s margin
    
    # Precompute normals for the 6 directions of the outer hexagon
    angles = np.deg2rad(np.arange(0, 360, 60))
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    # Precompute unit hex vertices (side 1)
    u_vertices = np.stack([np.cos(np.deg2rad(np.arange(0, 360, 60))), 
                           np.sin(np.deg2rad(np.arange(0, 360, 60)))], axis=1)
    
    def get_vertices(c, theta):
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        verts = u_vertices @ R.T + c
        return verts

    def compute_metrics(params):
        # params: 3N array
        centers = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        c = np.stack([centers, ys], axis=1)
        
        all_verts = []
        for i in range(N):
            v = get_vertices(c[i], thetas[i])
            all_verts.append(v)
        
        all_verts = np.vstack(all_verts)
        
        # Compute bounding box widths
        projections = all_verts @ normals.T
        max_p = np.max(projections, axis=0)
        min_p = np.min(projections, axis=0)
        widths = max_p - min_p
        max_width = np.max(widths)
        L = max_width / np.sqrt(3)
        
        # Compute penalty for overlaps
        penalty = 0.0
        thresh = np.sqrt(3)/2.0
        for i in range(N):
            vi = all_verts[i*6:(i+1)*6]
            ci = c[i]
            ti = thetas[i]
            R_i = np.array([[np.cos(ti), -np.sin(ti)], [np.sin(ti), np.cos(ti)]])
            
            for j in range(i+1, N):
                vj = all_verts[j*6:(j+1)*6]
                cj = c[j]
                tj = thetas[j]
                R_j = np.array([[np.cos(tj), -np.sin(tj)], [np.sin(tj), np.cos(tj)]])
                
                # Check vi inside hj
                v_rel = vi - cj
                p_local = v_rel @ R_j.T
                dists = p_local @ normals.T
                max_dists = np.max(dists, axis=1)
                over = np.maximum(0, max_dists - thresh)
                penalty += np.sum(over**2)
                
                # Check vj inside hi
                v_rel2 = vj - ci
                p_local2 = v_rel2 @ R_i.T
                dists2 = p_local2 @ normals.T
                max_dists2 = np.max(dists2, axis=1)
                over2 = np.maximum(0, max_dists2 - thresh)
                penalty += np.sum(over2**2)
                
        return L, penalty

    # Initialization
    init_params = np.zeros(3 * N)
    idx = 0
    for r in range(3):
        count = 3 if r == 0 else 4
        offset = 0.5 * np.sqrt(3) * (1 if r > 0 else 0)
        for c in range(count):
            x = c * np.sqrt(3) + offset
            y = r * 1.5
            theta = 0.0
            init_params[idx] = x
            init_params[idx+1] = y
            init_params[idx+2] = theta
            idx += 3
            
    com_x = np.mean(init_params[0::3])
    com_y = np.mean(init_params[1::3])
    init_params[0::3] -= com_x
    init_params[1::3] -= com_y
    
    best_params = init_params
    best_L, best_pen = compute_metrics(init_params)
    
    def obj_func(p):
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
        L, pen = compute_metrics(p)
        weight = 1000.0
        return L + weight * pen
        
    try:
        res = minimize(obj_func, init_params, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-10})
        if res.success:
            best_params = res.x
            best_L, best_pen = compute_metrics(best_params)
    except TimeoutError:
        pass
    except Exception:
        pass
        
    # Final metrics
    L, pen = compute_metrics(best_params)
    
    centers = best_params[0::3]
    ys = best_params[1::3]
    thetas = best_params[2::3]
    c = np.stack([centers, ys], axis=1)
    
    all_verts = []
    for i in range(N):
        v = get_vertices(c[i], thetas[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    projections = all_verts @ normals.T
    max_p = np.max(projections, axis=0)
    min_p = np.min(projections, axis=0)
    mid_p = (max_p + min_p) / 2.0
    
    n0 = normals[0]
    n1 = normals[1]
    m0 = mid_p[0]
    m1 = mid_p[1]
    
    A = np.array([n0, n1])
    b = np.array([m0, m1])
    try:
        C = np.linalg.solve(A, b)
    except:
        C = np.array([0.0, 0.0])
        
    widths = max_p - min_p
    max_width = np.max(widths)
    side = max_width / np.sqrt(3)
    
    inner = []
    for i in range(N):
        inner.append([best_params[i*3], best_params[i*3+1], np.rad2deg(best_params[i*3+2])])
        
    return inner, (C[0], C[1]), side, 0.0
# EVOLVE-BLOCK-END
