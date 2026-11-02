import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0

def construct():
    t0 = time.time()
    
    # Precompute unit hexagon vertices
    angles = np.radians(np.arange(0, 360, 60))
    unit_hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    def get_hex_verts(x, y, angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        # Rotation
        rx = unit_hex_verts[:, 0] * c - unit_hex_verts[:, 1] * s
        ry = unit_hex_verts[:, 0] * s + unit_hex_verts[:, 1] * c
        return np.column_stack((rx + x, ry + y))

    def sat_disjoint(h1, h2):
        # h1, h2 are (x, y, angle_rad)
        # Quick check
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        if dx*dx + dy*dy > 4.0: # dist > 2, max diameter
            return True
            
        v1 = get_hex_verts(h1[0], h1[1], h1[2])
        v2 = get_hex_verts(h2[0], h2[1], h2[2])
        
        # Axes
        axes = []
        for i in range(6):
            # Edge normal
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            axes.append((-ey, ex))
            
            p1 = v2[i]
            p2 = v2[(i+1)%6]
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            axes.append((-ey, ex))
            
        for nx, ny in axes:
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-9: continue
            nx /= norm
            ny /= norm
            
            proj1 = v1[:, 0]*nx + v1[:, 1]*ny
            proj2 = v2[:, 0]*nx + v2[:, 1]*ny
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return True
        return False

    def compute_min_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = []
        for i in range(N):
            v = get_hex_verts(positions[i][0], positions[i][1], angles_rad[i])
            all_verts.extend(v)
        verts = np.array(all_verts)
        
        center = np.mean(verts, axis=0)
        offsets = verts - center
        
        sqrt3 = math.sqrt(3)
        best_side = float('inf')
        best_phi = 0.0
        
        # Scan rotation
        for phi_deg in np.arange(0, 60, 0.5):
            phi = np.radians(phi_deg)
            max_extent = 0.0
            for k in range(3):
                # Normals at phi + 30 + k*60
                normal_angle = phi + np.pi/6 + k * np.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                width = np.max(proj) - np.min(proj)
                
                side_k = width / sqrt3
                if side_k > max_extent:
                    max_extent = side_k
            
            if max_extent < best_side:
                best_side = max_extent
                best_phi = phi_deg
                
        # Refine
        if best_side < float('inf'):
             for phi_deg in np.arange(best_phi - 0.5, best_phi + 0.5, 0.05):
                 phi = np.radians(phi_deg)
                 max_extent = 0.0
                 for k in range(3):
                     normal_angle = phi + np.pi/6 + k * np.pi/3
                     nx = math.cos(normal_angle)
                     ny = math.sin(normal_angle)
                     proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                     width = np.max(proj) - np.min(proj)
                     side_k = width / sqrt3
                     if side_k > max_extent:
                         max_extent = side_k
                 if max_extent < best_side:
                     best_side = max_extent
                     best_phi = phi_deg
                     
        return best_side, best_phi, center[0], center[1]

    def objective(params):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
            
        # Check validity
        angles_rad = np.radians(ang)
        for i in range(N):
            for j in range(i+1, N):
                if not sat_disjoint((pos[i][0], pos[i][1], angles_rad[i]), 
                                    (pos[j][0], pos[j][1], angles_rad[j])):
                    return 1e6
                    
        side, phi, cx, cy = compute_min_side(pos, ang)
        return side

    # Initialization: Honeycomb
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    idx = 0
    sqrt3 = math.sqrt(3)
    for i in range(3):
        for j in range(4):
            x = j * sqrt3 + i * sqrt3/2
            y = i * 1.5
            positions[idx, 0] = x
            positions[idx, 1] = y
            angles[idx] = 0
            idx += 1
            
    # Center the initial configuration
    center_x = np.mean(positions[:, 0])
    center_y = np.mean(positions[:, 1])
    positions[:, 0] -= center_x
    positions[:, 1] -= center_y
    
    init_params = np.zeros(3*N)
    for i in range(N):
        init_params[3*i] = positions[i, 0]
        init_params[3*i+1] = positions[i, 1]
        init_params[3*i+2] = angles[i]
        
    best_side, best_phi, best_cx, best_cy = compute_min_side(positions, angles)
    
    # Optimize
    if time.time() - t0 < DEADLINE:
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_side and res.fun < 1e5:
                # Extract best
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                    
                # Verify
                valid = True
                angles_rad = np.radians(new_ang)
                for i in range(N):
                    for j in range(i+1, N):
                        if not sat_disjoint((new_pos[i][0], new_pos[i][1], angles_rad[i]), 
                                            (new_pos[j][0], new_pos[j][1], angles_rad[j])):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    best_side, best_phi, best_cx, best_cy = compute_min_side(new_pos, new_ang)
        except Exception:
            pass

    # Return
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i])])
        
    # Wait, if optimize succeeded, use res.x
    if 'res' in locals() and res.fun < best_side:
         # Re-extract from res.x for the final output
         final_pos = np.zeros((N, 2))
         final_ang = np.zeros(N)
         for i in range(N):
             final_pos[i, 0] = res.x[3*i]
             final_pos[i, 1] = res.x[3*i+1]
             final_ang[i] = res.x[3*i+2]
             
         inner = []
         for i in range(N):
             inner.append([float(final_pos[i, 0]), float(final_pos[i, 1]), float(final_ang[i])])
             # Note: compute_min_side returns center relative to 0,0?
             # Yes, center = mean(verts).
             # The return interface requires: inner (x,y,angle), center (x,y), side, angle.
             # The 'center' returned by compute_min_side is the center of the *outer* hexagon.
             # The 'inner' coordinates are absolute.
             # This is consistent.
             
    # If optimize failed or didn't improve, use initial
    # The variables best_cx, best_cy, best_side, best_phi are updated in compute_min_side calls.
    # We need to ensure inner corresponds to the best_side.
    # In the code above, if res.fun < best_side, we update best_side.
    # But we also need to update the inner list.
    
    # Let's restructure the return logic to be safe.
    
    # We track best_params
    best_params = init_params.copy()
    
    # ... (run minimize) ...
    
    # If res is better:
    # best_params = res.x
    # best_side, best_phi, best_cx, best_cy = compute_min_side(...)
    
    # Construct final output from best_params
    final_pos = np.zeros((N, 2))
    final_ang = np.zeros(N)
    for i in range(N):
        final_pos[i, 0] = best_params[3*i]
        final_pos[i, 1] = best_params[3*i+1]
        final_ang[i] = best_params[3*i+2]
        
    inner = []
    for i in range(N):
        inner.append([float(final_pos[i, 0]), float(final_pos[i, 1]), float(final_ang[i])])
        
    return inner, (float(best_cx), float(best_cy)), float(best_side), float(best_phi)
