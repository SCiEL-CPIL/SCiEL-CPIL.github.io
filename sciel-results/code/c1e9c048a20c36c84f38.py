import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    deadline = time.time() + 140.0
    
    # 1. Initial Configuration
    # Lattice vectors
    d = math.sqrt(3) + 0.01
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    centers = []
    # Ring 0
    centers.append(np.array([0.0, 0.0]))
    # Ring 1
    ring1 = [
        u, v, v-u, -u, -v, u-v
    ]
    for vec in ring1:
        centers.append(vec.copy())
        
    # Ring 2 Candidates
    basis = [u, v, v-u, -u, -v, u-v]
    ring2_candidates = []
    for i in range(6):
        p1 = ring1[i]
        for b in basis:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
            is_r1 = False
            for p1j in ring1:
                if np.allclose(p2, p1j):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
                    
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    for i in range(5):
        centers.append(ring2_candidates[i].copy())
        
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    phi = 0.0
    
    x0 = np.zeros(27)
    for i in range(12):
        x0[2*i] = centers[i][0]
        x0[2*i+1] = centers[i][1]
    x0[24] = cx
    x0[25] = cy
    x0[26] = phi
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        c_centers = []
        for i in range(12):
            c_centers.append(np.array([params[2*i], params[2*i+1]]))
        ox = params[24]
        oy = params[25]
        o_phi = params[26]
        
        min_dist_sq = (math.sqrt(3) + 0.005)**2
        for i in range(12):
            for j in range(i+1, 12):
                diff = c_centers[i] - c_centers[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < min_dist_sq:
                    return 1e10 + (min_dist_sq - dist_sq)
                    
        verts = []
        for c in c_centers:
            for k in range(6):
                theta = math.radians(30.0) + k * math.pi / 3
                vx = c[0] + math.cos(theta)
                vy = c[1] + math.sin(theta)
                verts.append((vx, vy))
        
        max_proj = 0.0
        for k in range(6):
            n_angle = o_phi + math.pi/6 + k * math.pi/3
            nx = math.cos(n_angle)
            ny = math.sin(n_angle)
            for vx, vy in verts:
                proj = (vx - ox) * nx + (vy - oy) * ny
                if proj > max_proj:
                    max_proj = proj
        
        apothem = max_proj
        s = 2.0 * apothem / math.sqrt(3)
        
        return s

    best_result = None
    best_score = objective(x0)
    best_result = x0
    
    try:
        # Primary optimization
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        
        if time.time() < deadline:
            final_score = objective(res.x)
            if final_score < best_score:
                best_score = final_score
                best_result = res.x
    except Exception:
        pass
        
    # Restarts
    restarts = 0
    while time.time() < deadline - 15 and restarts < 3:
        x_perturbed = x0 + np.random.normal(0, 0.2, 27)
        try:
            res = minimize(objective, x_perturbed, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if time.time() > deadline - 10:
                break
            final_score = objective(res.x)
            if final_score < best_score:
                best_score = final_score
                best_result = res.x
        except Exception:
            pass
        restarts += 1
        
    inner = []
    for i in range(12):
        inner.append([best_result[2*i], best_result[2*i+1], 30.0])
        
    outer_center = (best_result[24], best_result[25])
    outer_side = best_score
    outer_angle = best_result[26]
    
    return inner, outer_center, outer_side, outer_angle
