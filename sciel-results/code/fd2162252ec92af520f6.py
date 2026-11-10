# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _hex_edge_normals(side, angle_deg):
    """Return 6 unit outward normals for a hexagon with given side and angle."""
    a = np.radians(angle_deg)
    # Edge normals are at angles a + 60k + 30 degrees (outward)
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

def _check_nonoverlap_fast(centers, angles_deg, side=1.0, tol=1e-6):
    """
    Vectorized SAT non-overlap check using Minkowski sums.
    Two convex polygons A and B overlap iff their Minkowski difference A-B
    contains the origin. For hexagons, this is equivalent to checking if
    the Minkowski sum of A and -B contains the origin.
    
    For two regular hexagons, the Minkowski sum is a 12-gon (or 6-gon if aligned).
    We check if origin is inside using the separating axis theorem:
    project onto all edge normals of both hexagons.
    """
    n = len(centers)
    if n < 2:
        return True
    
    # Precompute vertices and normals for all hexagons
    all_verts = np.zeros((n, 6, 2))
    all_normals = np.zeros((n, 6, 2))
    
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        all_verts[i] = verts
        normals = _hex_edge_normals(side, angles_deg[i])
        all_normals[i] = normals
    
    # For each pair, check SAT
    for i in range(n):
        for j in range(i+1, n):
            # Check separation using normals of hex i
            # Project both hexagons onto each normal of hex i
            v1 = all_verts[i]  # (6, 2)
            v2 = all_verts[j]  # (6, 2)
            n1 = all_normals[i]  # (6, 2)
            n2 = all_normals[j]  # (6, 2)
            
            # Project v1 and v2 onto normals of i
            proj1_i = v1 @ n1.T  # (6, 6)
            proj2_i = v2 @ n1.T  # (6, 6)
            min1_i = np.min(proj1_i, axis=0)
            max1_i = np.max(proj1_i, axis=0)
            min2_i = np.min(proj2_i, axis=0)
            max2_i = np.max(proj2_i, axis=0)
            
            separated = (max1_i + tol < min2_i) | (max2_i + tol < min1_i)
            if np.any(separated):
                continue
            
            # Project v1 and v2 onto normals of j
            proj1_j = v1 @ n2.T  # (6, 6)
            proj2_j = v2 @ n2.T  # (6, 6)
            min1_j = np.min(proj1_j, axis=0)
            max1_j = np.max(proj1_j, axis=0)
            min2_j = np.min(proj2_j, axis=0)
            max2_j = np.max(proj2_j, axis=0)
            
            separated = (max1_j + tol < min2_j) | (max2_j + tol < min1_j)
            if np.any(separated):
                continue
            
            # Overlap!
            return False
    
    return True

def _compute_all_vertices(centers, angles_deg, side=1.0):
    verts = []
    for i in range(len(centers)):
        verts.append(_hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i]))
    return np.vstack(verts)

def _initial_honeycomb():
    """
    Create a compact honeycomb arrangement of 12 hexagons.
    Optimal packing: 1 center + 6 first ring + 5 second ring.
    """
    d = np.sqrt(3.0)  # distance between adjacent hex centers
    positions = []
    angles = []
    
    # Center hexagon
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # First ring: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    # Second ring: 5 hexagons at 120-degree intervals
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
    """
    Alternative: try different second-ring placement.
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
    
    # Second ring: 5 hexagons, offset by 30 degrees
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

def _soft_min_outer_side(vertices_all, outer_angle_deg=0.0, beta=100.0):
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T  # shape (72, 6)
    
    beta_dots = beta * dots
    max_beta_dots = np.max(beta_dots)
    if max_beta_dots > 500:
        return (2.0 / np.sqrt(3.0)) * (max_beta_dots / beta)
    soft_max = max_beta_dots / beta + np.log(np.sum(np.exp(beta_dots - max_beta_dots))) / beta
    
    if soft_max < 0:
        soft_max = 0.0
    
    return (2.0 / np.sqrt(3.0)) * soft_max

def _smooth_objective(params, side=1.0, outer_angle_deg=0.0, beta=100.0):
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg, side)
    s = _soft_min_outer_side(all_verts, outer_angle_deg, beta)
    
    if not _check_nonoverlap_fast(centers, angles_deg, side):
        return s + 100.0
    
    return s

def _scipy_polish(params, deadline, outer_angle_deg=0.0, beta=100.0):
    def obj(x):
        return _smooth_objective(x, outer_angle_deg=outer_angle_deg, beta=beta)
    
    try:
        result = minimize(obj, params, method='L-BFGS-B', 
                          options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result.fun < _smooth_objective(params, outer_angle_deg=outer_angle_deg, beta=beta):
            c_pol = result.x.reshape(12, 3)[:, :2]
            a_pol = result.x.reshape(12, 3)[:, 2]
            if _check_nonoverlap_fast(c_pol, a_pol):
                return result.x
    except Exception:
        pass
    return params

def _local_refinement(params, deadline, outer_angle_deg=0.0):
    centers = params.reshape(12, 3)[:, :2]
    angles_deg = params.reshape(12, 3)[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts, outer_angle_deg)
    
    best_params = params.copy()
    best_s = s
    
    # Phase 1: Simulated annealing
    np.random.seed(42)
    T = 0.03
    cooling = 0.9995
    step_size = 0.03
    
    current_params = params.copy()
    current_s = s
    
    max_iters = 50000
    for it in range(max_iters):
        if time.time() > deadline - 60:
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
            step_size *= 0.92
    
    # Phase 2: Scipy polish
    if time.time() < deadline - 30:
        polished = _scipy_polish(best_params, deadline, outer_angle_deg, beta=100.0)
        c_pol = polished.reshape(12, 3)[:, :2]
        a_pol = polished.reshape(12, 3)[:, 2]
        if _check_nonoverlap_fast(c_pol, a_pol):
            verts_pol = _compute_all_vertices(c_pol, a_pol)
            s_pol = _min_outer_side(verts_pol, outer_angle_deg)
            if s_pol < best_s:
                best_s = s_pol
                best_params = polished
    
    # Phase 3: Fine-tuning
    if time.time() < deadline - 10:
        np.random.seed(123)
        T = 0.005
        cooling = 0.9998
        step_size = 0.008
        
        current_params = best_params.copy()
        current_s = best_s
        
        for it in range(30000):
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
            if it % 1000 == 0:
                step_size *= 0.95
    
    return best_params, best_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin before 630s kill
    
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
    
    # Global search: Simulated annealing with larger steps
    np.random.seed(42)
    T = 0.2
    cooling = 0.9997
    step_size = 0.15
    
    current_params = best_params.copy()
    current_s = best_s
    
    # Run until deadline minus reserved time for local refinement
    max_iters = 200000
    for it in range(max_iters):
        if time.time() > deadline - 180:  # Reserve 180s for local refinement
            break
        
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
        
        if it % 2000 == 0 and it > 0:
            step_size *= 0.93
            step_size = np.clip(step_size, 0.03, 0.4)
    
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
    
    if not _check_nonoverlap_fast(centers_final, angles_final):
        # Fallback to initial if somehow invalid
        centers_final, angles_final = _initial_honeycomb()
        all_verts_final = _compute_all_vertices(centers_final, angles_final)
        s_final = _min_outer_side(all_verts_final)
        best_outer_angle = 0.0
    
    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
