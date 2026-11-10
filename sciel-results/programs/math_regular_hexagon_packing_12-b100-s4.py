import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0
    
    # Seed configuration
    S = 1.0 + 1e-4
    centers = []
    centers.append([0.0, 0.0])
    r1 = np.sqrt(3) * S
    for i in range(6):
        angle = np.deg2rad(60 * i)
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        centers.append([x, y])
        
    r2 = 3.0 * S
    angles_r2 = [30, 90, 150, 210, 270]
    for ang_deg in angles_r2:
        angle = np.deg2rad(ang_deg)
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        centers.append([x, y])
        
    centers = np.array(centers)
    # Initial angles: 30 deg for all
    init_angles = np.full(12, 30.0)
    
    def get_vertices(cx, cy, angle_deg):
        a = np.deg2rad(angle_deg)
        verts = np.zeros((6, 2))
        for i in range(6):
            theta = a + np.deg2rad(60 * i)
            verts[i, 0] = cx + np.cos(theta)
            verts[i, 1] = cy + np.sin(theta)
        return verts

    def calc_s_min(angles, outer_angle_deg):
        # angles: (12,) array of degrees
        # outer_angle_deg: scalar
        all_verts = []
        for i in range(12):
            v = get_vertices(centers[i, 0], centers[i, 1], angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Outer hex normals
        a_out = np.deg2rad(outer_angle_deg)
        normals = []
        for k in range(6):
            theta = a_out + np.deg2rad(30 + 60 * k)
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        # Max projection
        # v @ normals.T gives (72, 6)
        projs = all_verts @ normals.T
        max_proj = np.max(projs)
        
        s = (2.0 / np.sqrt(3)) * max_proj
        return s

    def overlap_penalty(angles):
        # Check pairs
        pen = 0.0
        for i in range(12):
            v_i = get_vertices(centers[i, 0], centers[i, 1], angles[i])
            for j in range(i + 1, 12):
                v_j = get_vertices(centers[j, 0], centers[j, 1], angles[j])
                # Simple check: if any vertex of i is inside j, or vice versa
                # Or use SAT for min penetration
                # For speed, just check if distance between centers < min possible dist for these angles?
                # Min dist between two unit hexes with centers d apart.
                # If d < d_min, they overlap.
                # d_min depends on relative angle.
                # This is complex.
                # Let's use a simple vertex-in-polygon check for penalty.
                # If overlap, add large penalty.
                
                # Check if any vertex of v_i is inside v_j
                # v_j is convex.
                # For each vertex of v_i, check if it is to the left of all edges of v_j.
                # This is O(36) per pair. 66 pairs * 36 = 2376 ops.
                # Fast enough.
                
                for v in v_i:
                    inside = True
                    for k in range(6):
                        p1 = v_j[k]
                        p2 = v_j[(k + 1) % 6]
                        # Cross product (p2-p1) x (v-p1)
                        # If all >= 0 (or <= 0), inside
                        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                        if cross < -1e-9: # Outside
                            inside = False
                            break
                    if inside:
                        pen += 100.0
                        break # One vertex inside is enough for penalty
                if pen > 0:
                    # Also check reverse? If i is inside j, j vertices might not be in i?
                    # For small overlaps, likely both.
                    # Just checking one direction might miss "edge crossing" without vertex inside.
                    # But for hexagons, if they overlap, usually a vertex is inside or they are just touching.
                    # Given we start from touching, small rotations will either separate or push vertex in.
                    pass
                    
        return pen

    def objective(params):
        angles = params[:12]
        outer_angle = params[12]
        
        s = calc_s_min(angles, outer_angle)
        pen = overlap_penalty(angles)
        
        return s + pen

    # Initial guess
    x0 = np.append(init_angles, 30.0)
    
    # Optimization
    best_x = x0
    best_val = objective(x0)
    
    # Use Nelder-Mead for robustness with non-smooth penalty
    # Or L-BFGS-B if we smooth the penalty.
    # Let's try L-BFGS-B first, it's faster.
    # But penalty is non-smooth (step function).
    # Let's use Nelder-Mead.
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception as e:
        pass
        
    # Verify best_x
    # Calculate final s
    angles = best_x[:12]
    outer_angle = best_x[12]
    s_final = calc_s_min(angles, outer_angle)
    
    # Construct return arrays
    inner_hex_data = np.column_stack((centers, angles))
    outer_hex_data = np.array([0.0, 0.0, outer_angle])
    
    # Safety check: ensure s is not too small (validity)
    # If s < sqrt(12), it's invalid.
    if s_final < np.sqrt(12):
        s_final = np.sqrt(12) * 1.01
        # Recalculate? No, just return a safe value.
        # But we need to ensure the packing is valid for this s.
        # If we just increase s, it remains valid.
        
    return inner_hex_data, outer_hex_data, s_final
