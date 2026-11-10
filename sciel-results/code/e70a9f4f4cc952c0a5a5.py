# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a_rad = np.deg2rad(a_deg)
    angles = a_rad + np.deg2rad(60 * np.arange(6))
    x = cx + s * np.cos(angles)
    y = cy + s * np.sin(angles)
    return np.column_stack((x, y))

def _get_normals(vertices):
    normals = np.zeros((6, 2))
    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        normals[i] = n
    return normals

def _check_separation(hex1, hex2):
    n1 = _get_normals(hex1)
    n2 = _get_normals(hex2)
    all_normals = np.vstack([n1, n2])
    norms = np.linalg.norm(all_normals, axis=1, keepdims=True)
    all_normals = all_normals / norms

    for n in all_normals:
        proj1 = hex1 @ n
        proj2 = hex2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False

def _check_all_separated(verts):
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_separation(verts[i], verts[j]):
                return False
    return True

def _required_outer_size(all_vertices, outer_center, outer_angle_deg):
    a_rad = np.deg2rad(outer_angle_deg)
    normal_angles = a_rad + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    rel_vertices = all_vertices - outer_center
    proj = rel_vertices @ normals.T
    max_proj = np.max(proj)
    if max_proj < 0:
        return 0.0
    S = (2.0 / np.sqrt(3.0)) * max_proj
    return S

def _check_containment(shifted_inner_verts, shifted_outer_verts):
    for iv in shifted_inner_verts:
        for p in iv:
            for k in range(6):
                p1 = shifted_outer_verts[k]
                p2 = shifted_outer_verts[(k + 1) % 6]
                cross_val = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross_val < -1e-6:
                    return False
    return True

def _validate(inner_hex_data, outer_hex_data, side):
    """Full validation check. Returns True if valid."""
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data)) and np.isfinite(side)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False

    inner_verts = []
    for i in range(11):
        cx, cy, ang = inner_hex_data[i]
        inner_verts.append(_hex_vertices(cx, cy, 1.0, ang))

    if not _check_all_separated(inner_verts):
        return False

    ox, oy, oang = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, side, oang)
    shifted_outer = outer_verts - np.array([ox, oy])
    shifted_inner = [v - np.array([ox, oy]) for v in inner_verts]

    if not _check_containment(shifted_inner, shifted_outer):
        return False

    for sv in shifted_inner:
        for k in range(6):
            d = np.linalg.norm(sv[(k+1)%6] - sv[k])
            if abs(d - 1.0) > 1e-8:
                return False

    return True

def _build_flower_plus_4():
    h = np.sqrt(3)
    centers = np.array([
        [0.0, 0.0],
        [h, 0.0],
        [h*np.cos(np.deg2rad(60)), h*np.sin(np.deg2rad(60))],
        [h*np.cos(np.deg2rad(120)), h*np.sin(np.deg2rad(120))],
        [h*np.cos(np.deg2rad(180)), h*np.sin(np.deg2rad(180))],
        [h*np.cos(np.deg2rad(240)), h*np.sin(np.deg2rad(240))],
        [h*np.cos(np.deg2rad(300)), h*np.sin(np.deg2rad(300))],
        [2*h*np.cos(np.deg2rad(30)), 2*h*np.sin(np.deg2rad(30))],
        [2*h*np.cos(np.deg2rad(150)), 2*h*np.sin(np.deg2rad(150))],
        [2*h*np.cos(np.deg2rad(210)), 2*h*np.sin(np.deg2rad(210))],
        [2*h*np.cos(np.deg2rad(330)), 2*h*np.sin(np.deg2rad(330))],
    ])
    return centers

def _build_compact_cluster():
    h = np.sqrt(3)
    # 3-4-3-1 arrangement with tighter spacing
    sp = 2.0  # minimum center distance for non-overlapping hexes
    centers = np.array([
        [0, 0], [sp, 0], [2*sp, 0],
        [sp/2, h], [3*sp/2, h], [-sp/2, h], [5*sp/2, h],
        [0, 2*h], [sp, 2*h], [2*sp, 2*h],
        [sp/2, 3*h]
    ])
    return centers

def _build_staggered_v2():
    h = np.sqrt(3)
    # Alternative staggered pattern
    sp = 2.0
    centers = np.array([
        [0, 0], [sp, 0], [-sp, 0],
        [sp/2, h], [-sp/2, h], [3*sp/2, h], [-3*sp/2, h],
        [0, 2*h], [sp, 2*h], [-sp, 2*h],
        [sp/2, 3*h]
    ])
    return centers

def _build_linear_chain():
    h = np.sqrt(3)
    # Linear chain with alternating offsets
    sp = 2.0
    centers = []
    for i in range(11):
        x = i * sp
        y = 0 if i % 2 == 0 else h
        centers.append([x, y])
    return np.array(centers)

def _build_2d_grid():
    h = np.sqrt(3)
    sp = 2.0
    centers = []
    for i in range(4):
        for j in range(3):
            x = i * sp + (j % 2) * sp/2
            y = j * h
            centers.append([x, y])
    # Add one more
    centers.append([4 * sp/2, 4 * h/2])
    return np.array(centers[:11])

def _optimize_rigid_cluster(centers, deadline):
    """Optimize outer hexagon placement and angle for a fixed cluster."""
    inner_verts = [_hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0) for i in range(11)]
    if not _check_all_separated(inner_verts):
        return None, None, None
    
    all_inner_vertices = np.vstack(inner_verts)
    centroid = np.mean(centers, axis=0)
    
    best_S = float('inf')
    best_params = None
    
    def obj_outer(params):
        ox, oy, oang = params
        S = _required_outer_size(all_inner_vertices, np.array([ox, oy]), oang)
        return S
    
    for ang0 in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        if time.time() > deadline:
            break
        x0 = [centroid[0], centroid[1], float(ang0)]
        try:
            res = minimize(obj_outer, x0, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is not None:
        return centers.copy(), best_params, best_S
    return None, None, None

def _full_objective(params, all_inner_verts_cache=None):
    """Objective for joint optimization of centers and outer hex."""
    centers = params[:22].reshape(11, 2)
    ox, oy, oang = params[22], params[23], params[24]
    
    verts = [_hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0) for i in range(11)]
    
    # Separation penalty
    sep_pen = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_separation(verts[i], verts[j]):
                sep_pen += 50.0
    
    all_v = np.vstack(verts)
    S = _required_outer_size(all_v, np.array([ox, oy]), oang)
    
    # Containment penalty
    outer_verts = _hex_vertices(ox, oy, S, oang)
    shifted_outer = outer_verts - np.array([ox, oy])
    cont_pen = 0.0
    for iv in verts:
        for p in iv:
            ps = p - np.array([ox, oy])
            for k in range(6):
                p1 = shifted_outer[k]
                p2 = shifted_outer[(k + 1) % 6]
                cross_val = (p2[0] - p1[0]) * (ps[1] - p1[1]) - (p2[1] - p1[1]) * (ps[0] - p1[0])
                if cross_val < 0:
                    cont_pen += 50.0
                    break
            else:
                continue
            break
    
    return S + sep_pen + cont_pen

def _joint_optimize(initial_centers, initial_outer, deadline):
    """Joint optimization of all centers and outer hex."""
    x0 = np.concatenate([initial_centers.flatten(), initial_outer])
    
    best_result = None
    best_val = float('inf')
    
    # Try multiple perturbations
    rng = np.random.RandomState(42)
    attempts = 5
    for attempt in range(attempts):
        if time.time() > deadline:
            break
        if attempt == 0:
            x_start = x0.copy()
        else:
            noise = rng.randn(25) * 0.1 * (attempt + 1) / attempts
            x_start = x0 + noise
        
        try:
            res = minimize(_full_objective, x_start, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is not None:
        new_centers = best_result[:22].reshape(11, 2)
        new_outer = best_result[22:25]
        
        # Validate
        inner_hex_data_test = np.zeros((11, 3))
        for i in range(11):
            inner_hex_data_test[i] = [new_centers[i, 0], new_centers[i, 1], 0.0]
        
        if _validate(inner_hex_data_test, new_outer, new_outer[2]):
            return new_centers, new_outer, new_outer[2]
    
    return None, None, None

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580.0

    configs = [
        _build_flower_plus_4(),
        _build_compact_cluster(),
        _build_staggered_v2(),
        _build_linear_chain(),
        _build_2d_grid(),
    ]
    
    best_S = float('inf')
    best_inner = None
    best_outer = None
    
    # Phase 1: Rigid cluster optimization
    for centers in configs:
        if time.time() > deadline - 60:
            break
        
        inner_verts = [_hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0) for i in range(11)]
        if not _check_all_separated(inner_verts):
            continue
        
        result_centers, result_outer, result_S = _optimize_rigid_cluster(centers, deadline - 60)
        if result_S is not None and result_S < best_S:
            best_S = result_S
            best_inner = result_centers
            best_outer = result_outer
    
    # Fallback if no valid rigid cluster found
    if best_inner is None:
        best_inner = configs[0]
        centroid = np.mean(best_inner, axis=0)
        best_outer = np.array([centroid[0], centroid[1], 0.0])
        inner_verts = [_hex_vertices(best_inner[i, 0], best_inner[i, 1], 1.0, 0.0) for i in range(11)]
        all_inner_vertices = np.vstack(inner_verts)
        best_S = _required_outer_size(all_inner_vertices, np.array([centroid[0], centroid[1]]), 0.0)
    
    # Phase 2: Joint optimization with multiple restarts
    if time.time() < deadline - 30:
        # Try joint optimization from best rigid result
        res_centers, res_outer, res_S = _joint_optimize(best_inner, best_outer, deadline - 30)
        if res_S is not None and res_S < best_S:
            best_S = res_S
            best_inner = res_centers
            best_outer = res_outer
    
    # Phase 3: Additional joint optimization from other configs
    if time.time() < deadline - 15:
        for centers in configs[:3]:
            if time.time() > deadline - 15:
                break
            
            inner_verts = [_hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0) for i in range(11)]
            if not _check_all_separated(inner_verts):
                continue
            
            centroid = np.mean(centers, axis=0)
            init_outer = np.array([centroid[0], centroid[1], 0.0])
            
            res_centers, res_outer, res_S = _joint_optimize(centers, init_outer, deadline - 15)
            if res_S is not None and res_S < best_S:
                best_S = res_S
                best_inner = res_centers
                best_outer = res_outer
    
    # Build final output
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [best_inner[i, 0], best_inner[i, 1], 0.0]
    outer_hex_data = best_outer
    
    # Final validation with safety margin
    if not _validate(inner_hex_data, outer_hex_data, best_S):
        best_S += 1e-4
        if not _validate(inner_hex_data, outer_hex_data, best_S):
            best_S += 1e-3
            if not _validate(inner_hex_data, outer_hex_data, best_S):
                # Fallback to known valid configuration
                best_inner = _build_flower_plus_4()
                centroid = np.mean(best_inner, axis=0)
                best_outer = np.array([centroid[0], centroid[1], 0.0])
                inner_verts = [_hex_vertices(best_inner[i, 0], best_inner[i, 1], 1.0, 0.0) for i in range(11)]
                all_inner_vertices = np.vstack(inner_verts)
                best_S = _required_outer_size(all_inner_vertices, np.array([centroid[0], centroid[1]]), 0.0) + 1e-4
                
                inner_hex_data = np.zeros((11, 3))
                for i in range(11):
                    inner_hex_data[i] = [best_inner[i, 0], best_inner[i, 1], 0.0]
                outer_hex_data = best_outer

    return inner_hex_data, outer_hex_data, best_S

# EVOLVE-BLOCK-END
