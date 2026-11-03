import numpy as np
import time

N = 12
DEADLINE = 130.0  # Safety margin before 150s

def construct():
    start_time = time.time()
    
    # Regular hexagon vertices for unit hexagon at angle 0
    hex_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    hex_vertices = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # Outer hexagon edge normals (flat-top, rotation 0)
    outer_normals = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    outer_apothem_factor = np.sqrt(3) / 2.0  # S * this = apothem
    
    def get_inner_vertices(cx, cy, angle_deg):
        """Get vertices of unit hexagon centered at (cx,cy) with rotation angle_deg."""
        theta = angle_deg * np.pi / 180.0
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        verts = hex_vertices @ rot.T
        return cx + verts[:, 0], cy + verts[:, 1]
    
    def outer_side_length(cx_arr, cy_arr, angle_arr):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        max_val = 0.0
        for i in range(N):
            vx, vy = get_inner_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
            for j in range(6):
                for k in range(6):
                    val = (vx[j] * outer_normals[k, 0] + vy[j] * outer_normals[k, 1]) / outer_apothem_factor
                    if val > max_val:
                        max_val = val
        return max_val
    
    def check_overlap(i, j, cx_arr, cy_arr, angle_arr):
        """Check if hexagons i and j overlap using separating axis theorem."""
        verts_i = get_inner_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        verts_j = get_inner_vertices(cx_arr[j], cy_arr[j], angle_arr[j])
        
        # Get all edge normals from both hexagons
        normals = []
        for verts in [verts_i, verts_j]:
            for k in range(6):
                p1 = verts[k]
                p2 = verts[(k+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                normal /= (np.linalg.norm(normal) + 1e-12)
                normals.append(normal)
        
        for normal in normals:
            proj_i = np.dot(verts_i, normal)
            proj_j = np.dot(verts_j, normal)
            min_i, max_i = proj_i.min(), proj_i.max()
            min_j, max_j = proj_j.min(), proj_j.max()
            if max_i < min_j or max_j < min_i:
                return False  # Separating axis found, no overlap
        return True  # Overlap
    
    def total_overlap_penalty(cx_arr, cy_arr, angle_arr):
        """Compute penalty for overlapping hexagons."""
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap(i, j, cx_arr, cy_arr, angle_arr):
                    # Estimate overlap amount
                    dist = np.sqrt((cx_arr[i]-cx_arr[j])**2 + (cy_arr[i]-cy_arr[j])**2)
                    penalty += max(0, 2.0 - dist)  # Simple penalty
        return penalty
    
    def objective(state):
        cx_arr = state[0:N]
        cy_arr = state[N:2*N]
        angle_arr = state[2*N:3*N]
        side = outer_side_length(cx_arr, cy_arr, angle_arr)
        pen = total_overlap_penalty(cx_arr, cy_arr, angle_arr)
        return side + 10.0 * pen
    
    # Initial arrangement: hexagonal pattern
    # Center + ring1 (6) + ring2 partial (5)
    # Ring 1 distance: 2 * apothem = sqrt(3) for aligned hexagons
    r1 = np.sqrt(3)  # ~1.732
    r2 = 2 * np.sqrt(3)  # ~3.464
    
    cx = np.zeros(N)
    cy = np.zeros(N)
    ang = np.zeros(N)
    
    # Hex 0: center
    cx[0], cy[0] = 0.0, 0.0
    
    # Ring 1: 6 hexagons at distance r1
    for k in range(6):
        angle = k * 60.0 * np.pi / 180.0
        cx[1+k] = r1 * np.cos(angle)
        cy[1+k] = r1 * np.sin(angle)
    
    # Ring 2: 5 more hexagons (partial ring at distance r2)
    # Place at angles 30, 90, 150, 210, 270 (offset by 30 from ring 1)
    for k in range(5):
        angle = (30 + k * 72) * np.pi / 180.0  # Spread them out
        idx = 7 + k
        cx[idx] = r2 * np.cos(angle)
        cy[idx] = r2 * np.sin(angle)
    
    state = np.concatenate([cx, cy, ang])
    best_state = state.copy()
    best_obj = objective(state)
    
    # Simulated annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    while T > min_T and (time.time() - start_time) < DEADLINE:
        # Perturb a random hexagon
        i = np.random.randint(N)
        new_state = state.copy()
        new_state[i] += np.random.normal(0, 0.3)       # cx
        new_state[N+i] += np.random.normal(0, 0.3)     # cy
        new_state[2*N+i] += np.random.normal(0, 10.0)  # angle
        
        new_obj = objective(new_state)
        delta = new_obj - best_obj
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            state = new_state
            if new_obj < best_obj:
                best_obj = new_obj
                best_state = new_state.copy()
        
        T *= cooling
    
    # Final polish with coordinate descent
    if (time.time() - start_time) < DEADLINE - 20:
        for _ in range(50):
            if (time.time() - start_time) > DEADLINE - 10:
                break
            improved = False
            for i in range(N):
                for var_idx in [i, N+i, 2*N+i]:
                    for delta in [0.05, -0.05, 0.01, -0.01]:
                        trial = best_state.copy()
                        trial[var_idx] += delta
                        obj = objective(trial)
                        if obj < best_obj:
                            best_state = trial
                            best_obj = obj
                            improved = True
            if not improved:
                break
    
    cx_final = best_state[0:N]
    cy_final = best_state[N:2*N]
    ang_final = best_state[2*N:3*N]
    side_final = outer_side_length(cx_final, cy_final, ang_final)
    
    inner = [[cx_final[i], cy_final[i], ang_final[i]] for i in range(N)]
    return inner, (0.0, 0.0), side_final, 0.0

