import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    start_time = time.time()
    timeout = 550.0

    n_inner = 12
    min_dist = np.sqrt(3)
    
    # Initial guess
    np.random.seed(42)
    # Use a hexagonal lattice initial guess
    d = np.sqrt(3)
    # 12 points in a tight cluster
    # (0,0)
    # 6 at dist d
    # 5 at dist 2d?
    
    # Let's just use random
    angles = np.random.uniform(0, 2*np.pi, n_inner)
    radii = np.random.uniform(0.5, 2.5, n_inner)
    x0 = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    s0 = 4.5
    
    def objective(vars):
        return vars[-1]

    def constraints(vars):
        centers = vars[:24].reshape(12, 2)
        s = vars[-1]
        
        cons = []
        
        # Centroid constraint (equality)
        # sum(centers) = 0
        cons.append(np.sum(centers[:, 0]))
        cons.append(np.sum(centers[:, 1]))
        
        # Separation
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(centers[i] - centers[j])
                cons.append(dist - min_dist)
        
        # Containment
        normals = []
        for k in range(6):
            angle = np.deg2rad(30 + 60*k)
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        limit = s * np.sqrt(3) / 2
        
        inner_verts_offsets = []
        for k in range(6):
            angle = np.deg2rad(60*k)
            inner_verts_offsets.append(np.array([np.cos(angle), np.sin(angle)]))
        
        for i in range(12):
            c = centers[i]
            for n in normals:
                max_proj = -np.inf
                for off in inner_verts_offsets:
                    proj = np.dot(c, n) + np.dot(off, n)
                    if proj > max_proj:
                        max_proj = proj
                cons.append(limit - max_proj)
                
        return cons

    # SLSQP requires equality constraints to be in a specific format or mixed with inequalities?
    # scipy.optimize.minimize supports a list of dicts.
    # {'type': 'eq', 'fun': ...}
    # {'type': 'ineq', 'fun': ...}
    
    # We have 2 equality, 66 ineq (dist), 72 ineq (containment)
    
    def eq_constraints(vars):
        centers = vars[:24].reshape(12, 2)
        return [np.sum(centers[:, 0]), np.sum(centers[:, 1])]
        
    def ineq_constraints(vars):
        centers = vars[:24].reshape(12, 2)
        s = vars[-1]
        cons = []
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(centers[i] - centers[j])
                cons.append(dist - min_dist)
        
        normals = []
        for k in range(6):
            angle = np.deg2rad(30 + 60*k)
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        limit = s * np.sqrt(3) / 2
        
        inner_verts_offsets = []
        for k in range(6):
            angle = np.deg2rad(60*k)
            inner_verts_offsets.append(np.array([np.cos(angle), np.sin(angle)]))
        
        for i in range(12):
            c = centers[i]
            for n in normals:
                max_proj = -np.inf
                for off in inner_verts_offsets:
                    proj = np.dot(c, n) + np.dot(off, n)
                    if proj > max_proj:
                        max_proj = proj
                cons.append(limit - max_proj)
        return cons

    best_res = None
    best_s = np.inf
    
    for restart in range(5):
        if time.time() - start_time > timeout:
            break
            
        if restart == 0:
            x0_curr = x0
            s0_curr = s0
        else:
            if best_res is not None and best_res.success:
                x0_curr = best_res.x[:24].reshape(12, 2)
                s0_curr = best_res.x[-1]
                noise = np.random.normal(0, 0.1, (12, 2))
                x0_curr = x0_curr + noise
            else:
                angles = np.random.uniform(0, 2*np.pi, n_inner)
                radii = np.random.uniform(0.5, 2.5, n_inner)
                x0_curr = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
                s0_curr = 4.5
            
        try:
            res = minimize(
                objective, 
                np.hstack([x0_curr.flatten(), [s0_curr]]), 
                method='SLSQP', 
                constraints=[
                    {'type': 'eq', 'fun': eq_constraints},
                    {'type': 'ineq', 'fun': ineq_constraints}
                ],
                bounds=[(None, None)] * 24 + [(3.4, 10.0)],
                options={'maxiter': 500, 'ftol': 1e-6}
            )
            if res.success and res.fun < best_s:
                best_s = res.fun
                best_res = res
        except Exception:
            pass

    if best_res is None or best_s > 8.0:
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
    else:
        centers = best_res.x[:24].reshape(12, 2)
        s = best_res.x[-1]
        
        inner_hex_data = np.zeros((12, 3))
        for i in range(12):
            inner_hex_data[i, 0] = centers[i, 0]
            inner_hex_data[i, 1] = centers[i, 1]
            inner_hex_data[i, 2] = 0.0
            
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = s

    return inner_hex_data, outer_hex_data, outer_hex_side_length
