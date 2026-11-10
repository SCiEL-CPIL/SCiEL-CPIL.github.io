# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _hex_edge_normals(side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + (np.arange(6) + 0.5) * np.pi / 3.0
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _min_outer_side(vertices_all, outer_angle_deg=0.0):
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = vertices_all @ m.T
    max_dot = np.max(dots)
    if max_dot < 0:
        max_dot = 0.0
    return (2.0 / np.sqrt(3.0)) * max_dot

def _soft_min_outer_side(vertices_all, outer_angle_deg=0.0, beta=80.0):
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = vertices_all @ m.T
    beta_dots = beta * dots
    max_beta_dots = np.max(beta_dots)
    if max_beta_dots > 500:
        return (2.0 / np.sqrt(3.0)) * (max_beta_dots / beta)
    soft_max = max_beta_dots / beta + np.log(np.sum(np.exp(beta_dots - max_beta_dots))) / beta
    if soft_max < 0:
        soft_max = 0.0
    return (2.0 / np.sqrt(3.0)) * soft_max

def _check_nonoverlap_fast(centers, angles_deg, side=1.0, tol=1e-6):
    n = len(centers)
    if n < 2:
        return True
    
    all_verts = np.zeros((n, 6, 2))
    all_normals = np.zeros((n, 6, 2))
    
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        all_verts[i] = verts
        normals = _hex_edge_normals(side, angles_deg[i])
        all_normals[i] = normals
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = all_verts[i]
            v2 = all_verts[j]
            n1 = all_normals[i]
            n2 = all_normals[j]
            
            proj1_i = v1 @ n1.T
            proj2_i = v2 @ n1.T
            min1_i = np.min(proj1_i, axis=0)
            max1_i = np.max(proj1_i, axis=0)
            min2_i = np.min(proj2_i, axis=0)
            max2_i = np.max(proj2_i, axis=0)
            
            separated = (max1_i + tol < min2_i) | (max2_i + tol < min1_i)
            if np.any(separated):
                continue
            
            proj1_j = v1 @ n2.T
            proj2_j = v2 @ n2.T
            min1_j = np.min(proj1_j, axis=0)
            max1_j = np.max(proj1_j, axis=0)
            min2_j = np.min(proj2_j, axis=0)
            max2_j = np.max(proj2_j, axis=0)
            
            separated = (max1_j + tol < min1_j) | (max2_j + tol < min1_j)
            if np.any(separated):
                continue
            
            return False
    
    return True

def _compute_all_vertices(centers, angles_deg, side=1.0):
    verts = []
    for i in range(len(centers)):
        verts.append(_hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i]))
    return np.vstack(verts)

def _initial_honeycomb():
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
        angle = (k * 72.0) * np.pi / 180.0
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    return positions, angles

def _initial_compact_v2():
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
        angle = (k * 72.0 + 36.0) * np.pi / 180.0
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    return positions, angles

def _smooth_objective(params, side=1.0, outer_angle_deg=0.0, beta=80.0):
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg, side)
    s = _soft_min_outer_side(all_verts, outer_angle_deg, beta)
    
    if not _check_nonoverlap_fast(centers, angles_deg, side):
        return s + 100.0
    
    return s

def _scipy_polish(params, deadline, outer_angle_deg=0.0, beta=80.0):
    def obj(x):
        return _smooth_objective(x, outer_angle_deg=outer_angle_deg, beta=beta)
    
    try:
        result = minimize(obj, params, method='L-BFGS-B', 
                          options={'maxiter': 800, 'maxls': 100, 'ftol': 1e-14, 'gtol': 1e-12})
        
        if result.fun < _smooth_objective(params, outer_angle_deg=outer_angle_deg, beta=beta):
            c_pol = result.x.reshape(12, 3)[:, :2]
            a_pol = result.x.reshape(12, 3)[:, 2]
            if _check_nonoverlap_fast(c_pol, a_pol):
                return result.x
    except Exception:
        pass
    return params

def _directed_sa_move(current_params, step_size, centers, angles_deg, side=1.0):
    """
    Move a single hexagon in a directed manner: try moving it toward reducing
    the outer hexagon size, or away from overlapping neighbors.
    """
    idx = np.random.randint(0, 12)
    new_params = current_params.copy()
    
    # Random direction with bias toward center
    direction = np.random.randn(2)
    if np.random.random() < 0.5:
        direction -= centers[idx] * 0.1  # bias toward center
    
    new_params[idx * 3: idx * 3 + 2] += direction * step_size
    
    # Small angle perturbation
    if np.random.random() < 0.3:
        new_params[idx * 3 + 2] += np.random.randn() * step_size * 0.5
    
    return new_params

def _local_refinement(params, deadline, outer_angle_deg=0.0):
    centers = params.reshape(12, 3)[:, :2]
    angles_deg = params.reshape(12, 3)[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts, outer_angle_deg)
    
    best_params = params.copy()
    best_s = s
    
    # Phase 1: Directed simulated annealing
    np.random.seed(42)
    T = 0.02
    cooling = 0.9996
    step_size = 0.02
    
    current_params = params.copy()
    current_s = s
    
    max_iters = 80000
    for it in range(max_iters):
        if time.time() > deadline - 60:
            break
        
        new_params = _directed_sa_move(current_params, step_size, centers, angles_deg)
        c_new = new_params.reshape(12, 3)[:, :2]
        a_new = new_params.reshape(12, 3)[:, 2]
        
        if not _check_nonoverlap_fast(c_new, a_new):
            continue
        
        verts_new = _compute_all_vertices(c_new, a_new)
        s_new = _min_outer_side(verts_new, outer_angle_deg)
        
        delta = s_new - current_s
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_params = new_params
            current_s = s_new
            if s_new < best_s:
                best_s = s_new
                best_params = new_params.copy()
        
        T *= cooling
        if it % 3000 == 0:
            step_size *= 0.90
    
    # Phase 2: Multiple L-BFGS-B restarts
    if time.time() < deadline - 30:
        n_restarts = 12
        for si in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            if si == 0:
                x0 = best_params.copy()
            elif si == 1:
                x0 = best_params + np.random.randn(36) * 0.01
            elif si == 2:
                x0 = best_params + np.random.randn(36) * 0.02
            elif si == 3:
                x0 = best_params + np.random.randn(36) * 0.03
            elif si == 4:
                x0 = best_params + np.random.randn(36) * 0.05
            elif si == 5:
                x0 = best_params + np.random.randn(36) * 0.08
            else:
                x0 = best_params + np.random.randn(36) * (0.05 + 0.02 * (si - 5))
            
            polished = _scipy_polish(x0, deadline, outer_angle_deg, beta=80.0)
            c_pol = polished.reshape(12, 3)[:, :2]
            a_pol = polished.reshape(12, 3)[:, 2]
            if _check_nonoverlap_fast(c_pol, a_pol):
                verts_pol = _compute_all_vertices(c_pol, a_pol)
                s_pol = _min_outer_side(verts_pol, outer_angle_deg)
                if s_pol < best_s:
                    best_s = s_pol
                    best_params = polished
    
    # Phase 3: Fine-tuning with very small steps
    if time.time() < deadline - 10:
        np.random.seed(123)
        T = 0.003
        cooling = 0.9999
        step_size = 0.005
        
        current_params = best_params.copy()
        current_s = best_s
        
        for it in range(40000):
            if time.time() > deadline - 5:
                break
            
            new_params = current_params + np.random.randn(36) * step_size
            c_new = new_params.reshape(12, 3)[:, :2]
            a_new = new_params.reshape(12, 3)[:, 2]
            
            if not _check_nonoverlap_fast(c_new, a_new):
                continue
            
            verts_new = _compute_all_vertices(c_new, a_new)
            s_new = _min_outer_side(verts_new, outer_angle_deg)
            
            delta = s_new - current_s
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_params = new_params
                current_s = s_new
                if s_new < best_s:
                    best_s = s_new
                    best_params = new_params.copy()
            
            T *= cooling
            if it % 2000 == 0:
                step_size *= 0.93
    
    return best_params, best_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s kill
    
    # Try multiple initial configurations
    best_params = None
    best_s = np.inf
    
    for init_func in [_initial_honeycomb, _initial_compact_v2]:
        centers, angles_deg = init_func()
        params = np.column_stack([centers, angles_deg]).flatten()
        
        if not _check_nonoverlap_fast(centers, angles_deg):
            scale = 1.02
            while not _check_nonoverlap_fast(centers * scale, angles_deg) and scale < 3.0:
                scale += 0.02
            centers = centers * scale
            params = np.column_stack([centers, angles_deg]).flatten()
        
        all_verts = _compute_all_vertices(centers, angles_deg)
        s = _min_outer_side(all_verts)
        
        if s < best_s:
            best_s = s
            best_params = params.copy()
    
    # Global search: Simulated annealing with directed moves
    np.random.seed(42)
    T = 0.15
    cooling = 0.9998
    step_size = 0.12
    
    current_params = best_params.copy()
    current_s = best_s
    
    max_iters = 300000
    for it in range(max_iters):
        if time.time() > deadline - 200:
            break
        
        # Use directed moves for efficiency
        if it % 3 == 0:
            centers_cur = current_params.reshape(12, 3)[:, :2]
            angles_cur = current_params.reshape(12, 3)[:, 2]
            new_params = _directed_sa_move(current_params, step_size, centers_cur, angles_cur)
        else:
            new_params = current_params + np.random.randn(36) * step_size
        
        c_new = new_params.reshape(12, 3)[:, :2]
        a_new = new_params.reshape(12, 3)[:, 2]
        
        if not _check_nonoverlap_fast(c_new, a_new):
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
        
        if it % 3000 == 0 and it > 0:
            step_size *= 0.92
            step_size = np.clip(step_size, 0.02, 0.3)
    
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
    
    for angle_deg in np.arange(0, 60, 0.5):
        s_candidate = _min_outer_side(all_verts_final, angle_deg)
        if s_candidate < s_final:
            s_final = s_candidate
            best_outer_angle = angle_deg
    
    if time.time() < deadline - 5:
        for angle_deg in np.arange(best_outer_angle - 0.5, best_outer_angle + 0.5, 0.01):
            s_candidate = _min_outer_side(all_verts_final, angle_deg)
            if s_candidate < s_final:
                s_final = s_candidate
                best_outer_angle = angle_deg
        
        for angle_deg in np.arange(best_outer_angle - 0.01, best_outer_angle + 0.01, 0.001):
            s_candidate = _min_outer_side(all_verts_final, angle_deg)
            if s_candidate < s_final:
                s_final = s_candidate
                best_outer_angle = angle_deg
    
    if not _check_nonoverlap_fast(centers_final, angles_final):
        centers_final, angles_final = _initial_honeycomb()
        all_verts_final = _compute_all_vertices(centers_final, angles_final)
        s_final = _min_outer_side(all_verts_final)
        best_outer_angle = 0.0
    
    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
