import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Generate local positions for a 3-4-3 honeycomb of unit hexagons
    # Flat-top hexagons (angle 30 deg)
    local_pos = []
    dx = SQRT3
    dy = 1.5
    shift = SQRT3 / 2
    
    for row in range(3):
        for col in range(4):
            x = col * dx
            if row % 2 == 1:
                x += shift
            y = row * dy
            local_pos.append([x, y])
            
    local_pos = np.array(local_pos)
    local_pos -= local_pos.mean(axis=0)
    
    # Pre-compute local vertices for a unit hexagon (flat-top, 30 deg)
    alpha_local = np.radians(30)
    angles = alpha_local + np.radians(np.arange(0, 360, 60))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    local_verts_unit = np.column_stack([cos_a, sin_a])
    
    # Outer hexagon normals (fixed rotation 0, flat-top)
    # Normals at 30, 90, 150, 210, 270, 330
    outer_normals_angles = np.radians(np.arange(30, 360, 60))
    outer_normals = np.column_stack([np.cos(outer_normals_angles), np.sin(outer_normals_angles)])
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        x, y, phi = params
        
        c = np.cos(phi)
        s_phi = np.sin(phi)
        rot_matrix = np.array([[c, -s_phi], [s_phi, c]])
        
        # Rotate local vertices
        local_verts_rot = local_verts_unit @ rot_matrix.T
        
        min_s = 0.0
        for i in range(N):
            center = local_pos[i] @ rot_matrix
            center = center + np.array([x, y])
            verts = center + local_verts_rot
            
            # Project onto outer normals
            proj = verts @ outer_normals.T
            max_proj = np.max(proj, axis=0)
            
            # Required s
            s_req = max_proj * 2 / SQRT3
            # Update min_s with max of s_req
            current_max = np.max(s_req)
            if current_max > min_s:
                min_s = current_max
                
        return min_s

    best_s = 1e10
    best_params = [0.0, 0.0, 0.0]
    
    # Try multiple initial rotations
    for phi_init in [0, np.pi/6, np.pi/3, np.pi/12, -np.pi/12, np.pi/4]:
        x0_try = [0.0, 0.0, phi_init]
        try:
            res = minimize(
                objective,
                x0_try,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x
        except Exception:
            pass
            
    x_opt, y_opt, phi_opt = best_params
    
    c = np.cos(phi_opt)
    s_phi = np.sin(phi_opt)
    rot_matrix = np.array([[c, -s_phi], [s_phi, c]])
    
    global_angle_deg = np.degrees(alpha_local + phi_opt)
    
    inner = []
    for i in range(N):
        center = local_pos[i] @ rot_matrix
        center = center + np.array([x_opt, y_opt])
        inner.append([center[0], center[1], global_angle_deg])
        
    outer_center = (0.0, 0.0)
    outer_side = best_s
    outer_angle_deg = 0.0
    
    return inner, outer_center, outer_side, outer_angle_deg
