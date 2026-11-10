# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0
    safety_margin = 15.0

    def get_verts_normals(pos):
        # pos: (11, 2)
        x = pos[:, 0]
        y = pos[:, 1]
        angles_v = np.radians(60 * np.arange(6))
        angles_n = np.radians(30 + 60 * np.arange(6))
        
        cv = np.cos(angles_v)
        sv = np.sin(angles_v)
        cn = np.cos(angles_n)
        sn = np.sin(angles_n)
        
        V = np.zeros((11, 6, 2))
        V[:, :, 0] = x[:, np.newaxis] + cv[np.newaxis, :]
        V[:, :, 1] = y[:, np.newaxis] + sv[np.newaxis, :]
        
        N = np.zeros((11, 6, 2))
        N[:, :, 0] = cn[np.newaxis, :]
        N[:, :, 1] = sn[np.newaxis, :]
        
        return V, N

    def calc_side(V, theta_deg):
        theta = np.radians(theta_deg)
        outer_normals_angles = theta + np.radians(30 + 60 * np.arange(6))
        outer_normals = np.column_stack([np.cos(outer_normals_angles), np.sin(outer_normals_angles)])
        
        V_flat = V.reshape(-1, 2)
        projs = V_flat @ outer_normals.T
        max_projs = np.max(projs, axis=0)
        apothem = np.max(max_projs)
        return 2.0 * apothem / math.sqrt(3)

    def calc_penalty(V, N):
        penalty = 0.0
        tol = 1e-6
        
        for i in range(11):
            for j in range(i + 1, 11):
                Vi = V[i]
                Vj = V[j]
                Ni = N[i]
                Nj = N[j]
                Nij = np.vstack([Ni, Nj])
                
                proj_i = Vi @ Nij.T
                max_i = np.max(proj_i, axis=0)
                min_i = np.min(proj_i, axis=0)
                
                proj_j = Vj @ Nij.T
                max_j = np.max(proj_j, axis=0)
                min_j = np.min(proj_j, axis=0)
                
                gap1 = min_j - max_i
                gap2 = min_i - max_j
                gaps = np.maximum(gap1, gap2)
                
                max_gap = np.max(gaps)
                
                if max_gap < tol:
                    penalty += (tol - max_gap) ** 2
                    
        return penalty

    def objective(params):
        x = params[:22].reshape(11, 2)
        theta = params[22]
        
        V, N = get_verts_normals(x)
        side = calc_side(V, theta)
        pen = calc_penalty(V, N)
        
        return side + 100.0 * pen

    def is_valid(V, N, tol=1e-6):
        for i in range(11):
            for j in range(i + 1, 11):
                Vi = V[i]
                Vj = V[j]
                Ni = N[i]
                Nj = N[j]
                Nij = np.vstack([Ni, Nj])
                
                proj_i = Vi @ Nij.T
                max_i = np.max(proj_i, axis=0)
                min_i = np.min(proj_i, axis=0)
                
                proj_j = Vj @ Nij.T
                max_j = np.max(proj_j, axis=0)
                min_j = np.min(proj_j, axis=0)
                
                gap1 = min_j - max_i
                gap2 = min_i - max_j
                gaps = np.maximum(gap1, gap2)
                
                if np.max(gaps) <= tol:
                    return False
        return True

    h = math.sqrt(3) / 2
    d = math.sqrt(3)
    
    configs = []
    
    # Config 1: 4-3-4 Grid
    coords = []
    for i in range(4):
        coords.append([i * d, 0])
    for i in range(3):
        coords.append([i * d + d/2, d])
    for i in range(4):
        coords.append([i * d, 2 * d])
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
    configs.append(np.array(coords))
    
    # Config 2: 7-hex flower + 4 extra
    coords = [[0,0]]
    for k in range(6):
        ang = np.radians(30 + 60 * k)
        coords.append([d * np.cos(ang), d * np.sin(ang)])
    for k in range(4):
        ang = np.radians(60 * k)
        coords.append([2 * d * np.cos(ang), 2 * d * np.sin(ang)])
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
    configs.append(np.array(coords))
    
    # Config 3: Linear 11
    coords = []
    for i in range(11):
        coords.append([i * d, 0])
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
    configs.append(np.array(coords))
    
    best_side = float('inf')
    best_inner = None
    best_outer = None

    for idx, config in enumerate(configs):
        if time.time() > deadline - safety_margin:
            break
            
        for theta_init in [0.0, 15.0]:
            if time.time() > deadline - safety_margin:
                break
                
            x0 = np.concatenate([config.flatten(), [theta_init]])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if time.time() > deadline - safety_margin:
                    continue
                
                x_opt = res.x
                pos_opt = x_opt[:22].reshape(11, 2)
                theta_opt = x_opt[22]
                
                V, N = get_verts_normals(pos_opt)
                
                if is_valid(V, N):
                    side = calc_side(V, theta_opt)
                    if side < best_side:
                        best_side = side
                        best_inner = np.column_stack([pos_opt, np.zeros(11)])
                        best_outer = np.array([0.0, 0.0, theta_opt])
                        
            except Exception:
                pass

    if best_inner is None:
        pos = configs[0]
        best_inner = np.column_stack([pos, np.zeros(11)])
        V, N = get_verts_normals(pos)
        best_side = calc_side(V, 0.0)
        best_outer = np.array([0.0, 0.0, 0.0])

    # Polish with Nelder-Mead if time permits
    if best_inner is not None and time.time() < deadline - safety_margin:
        x0 = np.concatenate([best_inner[:, :2].flatten(), [best_outer[2]]])
        try:
            res_nm = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            x_opt = res_nm.x
            pos_opt = x_opt[:22].reshape(11, 2)
            theta_opt = x_opt[22]
            V, N = get_verts_normals(pos_opt)
            if is_valid(V, N):
                side = calc_side(V, theta_opt)
                if side < best_side:
                    best_side = side
                    best_inner = np.column_stack([pos_opt, np.zeros(11)])
                    best_outer = np.array([0.0, 0.0, theta_opt])
        except Exception:
            pass

    return best_inner, best_outer, best_side
# EVOLVE-BLOCK-END
