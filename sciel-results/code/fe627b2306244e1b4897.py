# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(vertices_all, outer_angle_deg=0.0):
    """
    Given array of all inner hexagon vertices (shape (72,2)), compute minimum outer hex side s
    (centered at origin, with given angle) that contains all vertices.
    """
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T
    max_dot = np.max(dots, axis=1)
    global_max = np.max(max_dot)
    
    if global_max < 0:
        global_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    """
    Check if all pairs of hexagons are non-overlapping using SAT.
    Returns True if no overlap, False otherwise.
    """
    n = len(centers)
    vertices_list = []
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        vertices_list.append(verts)
    
    normals_list = []
    for verts in vertices_list:
        normals = []
        for e in range(6):
            p1 = verts[e]
            p2 = verts[(e+1) % 6]
            edge = p2 - p1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append(np.array([nx/norm, ny/norm]))
        normals_list.append(normals)
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            overlap = True
            for hex_idx in range(2):
                verts = [v1, v2][hex_idx]
                normals = normals_list[hex_idx]
                for normal in normals:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 + tol < min2 or max2 + tol < min1:
                        overlap = False
                        break
                if not overlap:
                    break
            if overlap:
                return False
    return True

def _compute_all_vertices(centers, angles_deg, side=1.0):
    """Compute all 72 vertices of 12 hexagons."""
    verts = []
    for i in range(len(centers)):
        verts.append(_hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i]))
    return np.vstack(verts)

def _initial_honeycomb():
    """
    Create initial honeycomb-like arrangement of 12 hexagons.
    """
    d = np.sqrt(3.0)
    positions = []
    angles = []
    
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    d2 = 2.0 * d
    for k in range(5):
        angle = (k * 72 + 30) * np.pi / 180
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    return positions, angles

def _objective(params, side=1.0, outer_angle_deg=0.0):
    """
    Objective: minimize outer hex side s, with penalty for overlaps.
    params: (36,) = [x1,y1,theta1, x2,y2,theta2, ..., x12,y12,theta12]
    """
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg, side)
    s = _min_outer_side(all_verts, outer_angle_deg)
    
    if not _check_nonoverlap(centers, angles_deg, side):
        return s + 50.0
    
    return s

def _gradient_descent(params, centers, angles_deg, s, learning_rate=0.01, n_steps=1000, deadline=None, outer_angle_deg=0.0):
    """
    Perform gradient descent on the positions to minimize s.
    We approximate the gradient by finite differences.
    """
    best_s = s
    best_params = params.copy()
    
    for step in range(n_steps):
        if deadline and time.time() > deadline:
            break
        
        # Compute gradient for each parameter
        grad = np.zeros(36)
        eps = 0.01
        
        for i in range(36):
            orig_val = params[i]
            params[i] = orig_val + eps
            c_plus = params.reshape(12, 3)[:, :2]
            a_plus = params.reshape(12, 3)[:, 2]
            
            verts_plus = _compute_all_vertices(c_plus, a_plus)
            s_plus = _min_outer_side(verts_plus, outer_angle_deg)
            if not _check_nonoverlap(c_plus, a_plus):
                s_plus += 50.0
            
            params[i] = orig_val - eps
            c_minus = params.reshape(12, 3)[:, :2]
            a_minus = params.reshape(12, 3)[:, 2]
            
            verts_minus = _compute_all_vertices(c_minus, a_minus)
            s_minus = _min_outer_side(verts_minus, outer_angle_deg)
            if not _check_nonoverlap(c_minus, a_minus):
                s_minus += 50.0
            
            params[i] = orig_val
            grad[i] = (s_plus - s_minus) / (2 * eps)
        
        # Update parameters
        params -= learning_rate * grad
        
        # Evaluate new s
        c_new = params.reshape(12, 3)[:, :2]
        a_new = params.reshape(12, 3)[:, 2]
        verts_new = _compute_all_vertices(c_new, a_new)
        s_new = _min_outer_side(verts_new, outer_angle_deg)
        
        if _check_nonoverlap(c_new, a_new):
            if s_new < best_s:
                best_s = s_new
                best_params = params.copy()
        
        # Adaptive learning rate
        if step % 50 == 0 and step > 0:
            learning_rate *= 0.95
    
    return best_params, best_s

def _local_refinement(params, deadline, outer_angle_deg=0.0):
    """
    Perform local refinement using gradient descent and simulated annealing.
    """
    centers = params.reshape(12, 3)[:, :2]
    angles_deg = params.reshape(12, 3)[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts, outer_angle_deg)
    
    # Phase 1: Gradient descent
    params, s = _gradient_descent(params, centers, angles_deg, s, 
                                   learning_rate=0.05, n_steps=2000, deadline=deadline,
                                   outer_angle_deg=outer_angle_deg)
    
    # Phase 2: Simulated annealing with smaller steps
    np.random.seed(42)
    T = 0.05
    cooling = 0.999
    step_size = 0.05
    
    current_params = params.copy()
    current_s = s
    
    for it in range(50000):
        if time.time() > deadline:
            break
        
        new_params = current_params + np.random.randn(36) * step_size
        c_new = new_params.reshape(12, 3)[:, :2]
        a_new = new_params.reshape(12, 3)[:, 2]
        
        if not _check_nonoverlap(c_new, a_new):
            continue
        
        verts_new = _compute_all_vertices(c_new, a_new)
        s_new = _min_outer_side(verts_new, outer_angle_deg)
        
        delta = s_new - current_s
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_params = new_params
            current_s = s_new
            if s_new < s:
                s = s_new
                params = new_params.copy()
        
        T *= cooling
        if it % 1000 == 0:
            step_size *= 0.9
    
    return params, s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s kill
    
    # Start with honeycomb arrangement
    centers, angles_deg = _initial_honeycomb()
    params = np.column_stack([centers, angles_deg]).flatten()
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    best_s = _min_outer_side(all_verts)
    best_params = params.copy()
    
    if not _check_nonoverlap(centers, angles_deg):
        scale = 1.1
        while not _check_nonoverlap(centers * scale, angles_deg) and scale < 3.0:
            scale += 0.05
        centers = centers * scale
        params = np.column_stack([centers, angles_deg]).flatten()
        all_verts = _compute_all_vertices(centers, angles_deg)
        best_s = _min_outer_side(all_verts)
        best_params = params.copy()
    
    # Global search: Simulated annealing with larger steps
    np.random.seed(42)
    T = 0.3
    cooling = 0.9995
    step_size = 0.2
    
    current_params = best_params.copy()
    current_s = best_s
    
    # Run until deadline or convergence
    max_iters = 300000
    for it in range(max_iters):
        if time.time() > deadline - 120:  # Reserve 120s for local refinement
            break
        
        new_params = current_params + np.random.randn(36) * step_size
        c_new = new_params.reshape(12, 3)[:, :2]
        a_new = new_params.reshape(12, 3)[:, 2]
        
        if not _check_nonoverlap(c_new, a_new):
            continue
        
        verts_new = _compute_all_vertices(c_new, a_new)
        s_new = _min_outer_side(verts_new)
        
        delta = s_new - current_s
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_params = new_params
            current_s = s_new
            if s_new < best_s:
                best_s = s_new
                best_params = new_params.copy()
        
        T *= cooling
        
        if it % 1000 == 0 and it > 0:
            step_size *= 0.95
            step_size = np.clip(step_size, 0.05, 0.5)
    
    # Local refinement
    best_params, best_s = _local_refinement(best_params, deadline)
    
    # Final verification and cleanup
    centers_final = best_params.reshape(12, 3)[:, :2]
    angles_final = best_params.reshape(12, 3)[:, 2]
    
    centroid = np.mean(centers_final, axis=0)
    centers_final = centers_final - centroid
    
    all_verts_final = _compute_all_vertices(centers_final, angles_final)
    
    # Search for best outer hexagon angle
    best_outer_angle = 0.0
    s_final = _min_outer_side(all_verts_final, 0.0)
    
    # Coarse search over outer angles
    for angle_deg in np.arange(0, 60, 0.5):
        s_candidate = _min_outer_side(all_verts_final, angle_deg)
        if s_candidate < s_final:
            s_final = s_candidate
            best_outer_angle = angle_deg
    
    # Fine search around best angle
    if time.time() < deadline - 5:
        for angle_deg in np.arange(best_outer_angle - 0.5, best_outer_angle + 0.5, 0.01):
            s_candidate = _min_outer_side(all_verts_final, angle_deg)
            if s_candidate < s_final:
                s_final = s_candidate
                best_outer_angle = angle_deg
        
        # Even finer search
        for angle_deg in np.arange(best_outer_angle - 0.01, best_outer_angle + 0.01, 0.001):
            s_candidate = _min_outer_side(all_verts_final, angle_deg)
            if s_candidate < s_final:
                s_final = s_candidate
                best_outer_angle = angle_deg
    
    if not _check_nonoverlap(centers_final, angles_final):
        # Fallback to initial if somehow invalid
        centers_final, angles_final = _initial_honeycomb()
        all_verts_final = _compute_all_vertices(centers_final, angles_final)
        s_final = _min_outer_side(all_verts_final)
        best_outer_angle = 0.0
    
    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
