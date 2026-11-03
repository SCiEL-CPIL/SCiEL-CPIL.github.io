import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Generate initial lattice points
    points_axial = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            dist_sq = q*q + q*r + r*r
            points_axial.append((dist_sq, q, r))
    
    points_axial.sort()
    lattice_points = [(q, r) for _, q, r in points_axial[:12]]
    
    pts = np.array([
        [np.sqrt(3) * (q + r/2.0), 1.5 * r] 
        for q, r in lattice_points
    ])
    pts = pts - np.mean(pts, axis=0)

    def evaluate_rigid(params):
        theta, xc, yc, phi = params
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        R_phi = np.array([[c_phi, -s_phi], [s_phi, c_phi]])
        rotated_pts = pts @ R_phi.T + np.array([xc, yc])
        
        max_proj = 0.0
        for k in range(6):
            angle = theta + k * np.pi / 3.0
            n = np.array([np.cos(angle), np.sin(angle)])
            projs = np.abs(rotated_pts @ n)
            max_proj = max(max_proj, np.max(projs))
        
        return (2.0 / np.sqrt(3.0)) * max_proj

    best_S = float('inf')
    best_params = None
    
    # Try a few random initializations for rigid optimization
    for _ in range(5):
        if time.time() > deadline:
            break
        x0 = [np.random.uniform(0, np.pi), np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(0, np.pi)]
        try:
            res = minimize(evaluate_rigid, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to seed-like behavior or simple estimate
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    if time.time() > deadline:
        theta, xc, yc, phi = best_params
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        R_phi = np.array([[c_phi, -s_phi], [s_phi, c_phi]])
        final_pts = (pts @ R_phi.T) + np.array([xc, yc])
        angle_deg = np.degrees(phi)
        inner = [[float(p[0]), float(p[1]), float(angle_deg)] for p in final_pts]
        return inner, (0.0, 0.0), float(best_S), float(np.degrees(theta))

    # Stage 2: Jitter
    def evaluate_jitter(params):
        theta, xc, yc, phi = params[0], params[1], params[2], params[3]
        p_coords = params[4:]
        p_pts = p_coords.reshape(12, 2)
        
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        R_phi = np.array([[c_phi, -s_phi], [s_phi, c_phi]])
        global_pts = (p_pts @ R_phi.T) + np.array([xc, yc])
        
        penalty = 0.0
        min_dist = np.sqrt(3.0)
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(global_pts[i] - global_pts[j])
                if d < min_dist:
                    penalty += (min_dist - d)**2
        
        max_proj = 0.0
        for k in range(6):
            angle = theta + k * np.pi / 3.0
            n = np.array([np.cos(angle), np.sin(angle)])
            projs = np.abs(global_pts @ n)
            max_proj = max(max_proj, np.max(projs))
        
        S = (2.0 / np.sqrt(3.0)) * max_proj
        return S + 100.0 * penalty

    # Initialize jitter from best rigid
    theta, xc, yc, phi = best_params
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    R_phi_inv = np.array([[c_phi, s_phi], [-s_phi, c_phi]])
    local_pts = (pts - np.array([xc, yc])) @ R_phi_inv.T
    p_init = local_pts.flatten().tolist()
    
    x0_jitter = list(best_params) + p_init
    
    try:
        res2 = minimize(evaluate_jitter, x0_jitter, method='Powell', options={'maxiter': 100, 'xtol': 1e-4, 'ftol': 1e-4})
        if res2.fun < best_S:
            best_S = res2.fun
            best_params = res2.x
    except Exception:
        pass

    theta, xc, yc, phi = best_params[0], best_params[1], best_params[2], best_params[3]
    if len(best_params) > 4:
        p_coords = best_params[4:]
        p_pts = p_coords.reshape(12, 2)
    else:
        p_pts = pts
        
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    R_phi = np.array([[c_phi, -s_phi], [s_phi, c_phi]])
    final_pts = (p_pts @ R_phi.T) + np.array([xc, yc])
    
    angle_deg = np.degrees(phi)
    inner = [[float(p[0]), float(p[1]), float(angle_deg)] for p in final_pts]
    
    return inner, (0.0, 0.0), float(best_S), float(np.degrees(theta))
