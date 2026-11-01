import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin from 150s limit
    
    # 1. Define relative centers of the 11 hexagons in a 3-4-3 triangular lattice
    d = np.sqrt(3)
    
    # Row 1 (middle, 4 hexagons) at y = -1.5
    centers_row1 = np.array([[-1.5*d, -1.5], [-0.5*d, -1.5], [0.5*d, -1.5], [1.5*d, -1.5]])
    
    # Row 0 (top, 3 hexagons) at y = 0
    centers_row0 = np.array([[-d, 0.0], [0.0, 0.0], [d, 0.0]])
    
    # Row 2 (bottom, 3 hexagons) at y = -3.0
    centers_row2 = np.array([[-d, -3.0], [0.0, -3.0], [d, -3.0]])
    
    centers = np.vstack([centers_row0, centers_row1, centers_row2])
    
    # Pre-compute unit vectors for hexagon vertices relative to center
    hex_angles = np.arange(6) * np.pi / 3
    hex_verts_rel = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # Pre-compute container normals relative to container orientation
    container_normal_angles_rel = np.arange(6) * np.pi / 3 + np.pi / 6
    
    def objective(params):
        alpha, phi = params
        
        # Container normals in global coords
        c_nor_angles = container_normal_angles_rel + phi
        c_normals = np.column_stack([np.cos(c_nor_angles), np.sin(c_nor_angles)])
        
        max_proj = 0.0
        for n in c_normals:
            # Project all centers onto n
            center_projs = centers.dot(n)
            
            # Project relative vertices onto n
            # v_rel dot R(-alpha) n
            cos_a = np.cos(-alpha)
            sin_a = np.sin(-alpha)
            n_rot_x = n[0]*cos_a - n[1]*sin_a
            n_rot_y = n[0]*sin_a + n[1]*cos_a
            
            vert_projs = hex_verts_rel[:, 0] * n_rot_x + hex_verts_rel[:, 1] * n_rot_y
            
            max_vert_proj = np.max(vert_projs)
            min_vert_proj = np.min(vert_projs)
            
            pos_extent = np.max(center_projs) + max_vert_proj
            neg_extent = np.min(center_projs) + min_vert_proj
            
            curr_max = max(pos_extent, -neg_extent)
            if curr_max > max_proj:
                max_proj = curr_max
                
        return (2.0 / np.sqrt(3.0)) * max_proj

    bounds = [(0, np.pi/6), (0, np.pi/6)]
    
    best_result = None
    best_val = float('inf')
    
    # Grid search
    n_steps = 30
    alphas = np.linspace(0, np.pi/6, n_steps)
    phis = np.linspace(0, np.pi/6, n_steps)
    
    for a in alphas:
        if time.time() > deadline:
            break
        for p in phis:
            val = objective([a, p])
            if val < best_val:
                best_val = val
                best_result = [a, p]
        
    # Local refinement
    if best_result is not None and time.time() < deadline - 5.0:
        try:
            res = minimize(objective, best_result, method='Nelder-Mead', bounds=bounds, 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass
            
    alpha_opt, phi_opt = best_result
    S_opt = best_val
    
    inner = []
    for cx, cy in centers:
        angle_deg = np.degrees(alpha_opt)
        inner.append([cx, cy, angle_deg])
        
    outer_center = (0.0, 0.0)
    side_length = S_opt
    outer_angle = np.degrees(phi_opt)
    
    return inner, outer_center, side_length, outer_angle
# EVOLVE-BLOCK-END
