# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _outer_side_length_from_positions(positions, angles_deg):
    """
    Compute the minimum side length of a regular hexagon (centered at origin, angle 0)
    that contains all the given unit hexagons.
    """
    n = len(positions)
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2
    max_proj = 0.0
    for i in range(n):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
    s = max_proj / apothem_factor
    return max(s, np.sqrt(12) - 1e-12)

def _check_validity(inner_hex_data, outer_hex_data, s, tol=1e-6):
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    # SAT check for non-overlap
    for i in range(12):
        for j in range(i + 1, 12):
            verts_i = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
            verts_j = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, inner_hex_data[j, 2])
            separated = False
            for k in range(6):
                edge = verts_i[(k + 1) % 6] - verts_i[k]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    normal = normal / norm
                    proj_i = verts_i @ normal
                    proj_j = verts_j @ normal
                    if np.max(proj_i) + tol < np.min(proj_j) or np.max(proj_j) + tol < np.min(proj_i):
                        separated = True
                        break
            if not separated:
                for k in range(6):
                    edge = verts_j[(k + 1) % 6] - verts_j[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-12:
                        normal = normal / norm
                        proj_i = verts_i @ normal
                        proj_j = verts_j @ normal
                        if np.max(proj_i) + tol < np.min(proj_j) or np.max(proj_j) + tol < np.min(proj_i):
                            separated = True
                            break
            if not separated:
                return False
    # Containment check
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -tol:
                    return False
    return True

def _objective(params, beta=100.0):
    """
    Smooth objective: minimize s (outer hex side) + penalties for overlap and containment.
    params: (36,) array of [x0, y0, a0, x1, y1, a1, ...] for 12 hexagons.
    """
    pos = np.column_stack([params[0::3], params[1::3]])
    ang = params[2::3]
    
    # Compute all vertex positions
    all_verts = np.zeros((12, 6, 2))
    for i in range(12):
        all_verts[i] = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
    
    # Outer hexagon normals (for angle 0)
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2
    
    # Log-sum-exp for smooth max projection
    flat_verts = all_verts.reshape(-1, 2)
    proj = flat_verts @ normals.T  # (72, 6)
    flat_proj = proj.reshape(-1)
    
    m = np.max(flat_proj)
    if m > 1e10:
        m = 1e10
    lse = m + np.log(np.sum(np.exp((flat_proj - m) * beta)) / beta)
    s_smooth = lse / apothem_factor
    s_smooth = max(s_smooth, np.sqrt(12) - 1e-12)
    
    # Overlap penalty using SAT
    overlap_pen = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            verts_i = all_verts[i]
            verts_j = all_verts[j]
            for k in range(6):
                edge_i = verts_i[(k + 1) % 6] - verts_i[k]
                normal_i = np.array([-edge_i[1], edge_i[0]])
                norm_i = np.linalg.norm(normal_i)
                if norm_i > 1e-12:
                    normal_i = normal_i / norm_i
                    proj_i = verts_i @ normal_i
                    proj_j = verts_j @ normal_i
                    pen = np.max(proj_i) - np.min(proj_j)
                    if pen > 0:
                        overlap_pen += pen * pen
            for k in range(6):
                edge_j = verts_j[(k + 1) % 6] - verts_j[k]
                normal_j = np.array([-edge_j[1], edge_j[0]])
                norm_j = np.linalg.norm(normal_j)
                if norm_j > 1e-12:
                    normal_j = normal_j / norm_j
                    proj_i = verts_i @ normal_j
                    proj_j = verts_j @ normal_j
                    pen = np.max(proj_i) - np.min(proj_j)
                    if pen > 0:
                        overlap_pen += pen * pen
    
    # Containment penalty
    s_for_contain = max(s_smooth, 2.0)
    outer_verts = _hex_vertices(0.0, 0.0, s_for_contain, 0.0)
    contain_pen = 0.0
    for i in range(12):
        for v in all_verts[i]:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < 0:
                    contain_pen += cross * cross
    
    return s_smooth + 100.0 * overlap_pen + 100.0 * contain_pen

def _simulated_annealing(start_params, start_temp=1.0, end_temp=0.01, n_iter=2000, time_limit=30.0):
    """
    Simulated annealing to minimize the objective.
    """
    start_time = time.time()
    current_params = start_params.copy()
    current_obj = _objective(current_params, beta=50.0)
    best_params = current_params.copy()
    best_obj = current_obj
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Temperature schedule
        t = i / n_iter
        temp = start_temp * (end_temp / start_temp) ** t
        
        # Generate neighbor: perturb one or two parameters
        n_perturb = 1 if np.random.rand() < 0.7 else 2
        new_params = current_params.copy()
        for _ in range(n_perturb):
            idx = np.random.randint(0, 36)
            if idx % 3 == 2:  # angle
                new_params[idx] += np.random.randn() * 0.1 * temp
            else:  # position
                new_params[idx] += np.random.randn() * 0.1 * temp
        
        new_obj = _objective(new_params, beta=50.0)
        delta = new_obj - current_obj
        
        if delta < 0 or np.random.rand() < np.exp(-delta / max(temp, 1e-10)):
            current_params = new_params
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_params = current_params.copy()
    
    return best_params, best_obj

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0
    sqrt3 = np.sqrt(3)
    
    # Generate diverse starting configurations
    starting_configs = []
    
    # 1. Honeycomb: center + ring of 6 + 5 more
    honeycomb_positions = np.array([
        [0, 0],
        [3, 0], [1.5, sqrt3], [-1.5, sqrt3],
        [-3, 0], [-1.5, -sqrt3], [1.5, -sqrt3],
        [4.5, sqrt3], [3, 2*sqrt3], [0, 2*sqrt3],
        [-3, 2*sqrt3], [-4.5, sqrt3]
    ])
    starting_configs.append((honeycomb_positions.copy(), np.zeros(12)))
    
    # 2. Rotated honeycombs
    for rot in [5, 10, 15, 20, 25, 30]:
        ang = np.full(12, rot)
        starting_configs.append((honeycomb_positions.copy(), ang))
    
    # 3. Perturbed honeycombs
    np.random.seed(42)
    for t in range(20):
        pos = honeycomb_positions + np.random.randn(12, 2) * 0.5
        ang = np.random.uniform(-30, 30, 12)
        starting_configs.append((pos, ang))
    
    # 4. Ring configuration: 6 inner, 6 outer
    inner_ring = np.array([
        [3, 0], [1.5, sqrt3], [-1.5, sqrt3],
        [-3, 0], [-1.5, -sqrt3], [1.5, -sqrt3]
    ])
    outer_ring = np.array([
        [4.5, 1.5*sqrt3], [0, 3*sqrt3], [-4.5, 1.5*sqrt3],
        [-4.5, -1.5*sqrt3], [0, -3*sqrt3], [4.5, -1.5*sqrt3]
    ])
    ring_pos = np.vstack([inner_ring, outer_ring])
    starting_configs.append((ring_pos, np.zeros(12)))
    
    # 5. Compact cluster
    cluster_pos = np.array([
        [0, 0], [2.8, 0], [1.4, 2.42],
        [-1.4, 2.42], [-2.8, 0], [-1.4, -2.42],
        [1.4, -2.42], [4.2, 2.42], [2.8, 4.85],
        [0, 4.85], [-2.8, 4.85], [-4.2, 2.42]
    ])
    starting_configs.append((cluster_pos, np.zeros(12)))
    
    best_params = None
    best_s = float('inf')
    
    for idx, (pos, ang) in enumerate(starting_configs):
        if time.time() > deadline - 30:
            break
        
        p = np.zeros(36)
        for i in range(12):
            p[3*i] = pos[i, 0]
            p[3*i+1] = pos[i, 1]
            p[3*i+2] = ang[i]
        
        # First: simulated annealing
        sa_params, sa_obj = _simulated_annealing(p, start_temp=0.5, end_temp=0.01, n_iter=1000, time_limit=15.0)
        
        # Then: L-BFGS-B polish
        if time.time() < deadline - 15:
            try:
                def obj_func(params):
                    return _objective(params, beta=100.0)
                
                res = minimize(obj_func, sa_params, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30})
                
                pos_opt = np.column_stack([res.x[0::3], res.x[1::3]])
                ang_opt = res.x[2::3]
                s_opt = _outer_side_length_from_positions(pos_opt, ang_opt)
                
                inner_hex_data = np.zeros((12, 3))
                for i in range(12):
                    inner_hex_data[i, 0] = res.x[3*i]
                    inner_hex_data[i, 1] = res.x[3*i+1]
                    inner_hex_data[i, 2] = res.x[3*i+2]
                outer_hex_data = np.array([0.0, 0.0, 0.0])
                
                if _check_validity(inner_hex_data, outer_hex_data, s_opt):
                    if s_opt < best_s:
                        best_s = s_opt
                        best_params = res.x.copy()
            except Exception:
                pass
        
        # Fine polish with higher beta if we have time
        if best_params is not None and time.time() < deadline - 5:
            try:
                def obj_func_fine(params):
                    return _objective(params, beta=300.0)
                
                res2 = minimize(obj_func_fine, best_params, method='L-BFGS-B',
                                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 20})
                
                pos_fine = np.column_stack([res2.x[0::3], res2.x[1::3]])
                ang_fine = res2.x[2::3]
                s_fine = _outer_side_length_from_positions(pos_fine, ang_fine)
                
                inner_hex_data = np.zeros((12, 3))
                for i in range(12):
                    inner_hex_data[i, 0] = res2.x[3*i]
                    inner_hex_data[i, 1] = res2.x[3*i+1]
                    inner_hex_data[i, 2] = res2.x[3*i+2]
                outer_hex_data = np.array([0.0, 0.0, 0.0])
                
                if _check_validity(inner_hex_data, outer_hex_data, s_fine):
                    if s_fine < best_s:
                        best_s = s_fine
                        best_params = res2.x.copy()
            except Exception:
                pass
    
    # Fallback if no valid solution found
    if best_params is None:
        best_params = np.zeros(36)
        for i in range(12):
            best_params[3*i] = honeycomb_positions[i, 0]
            best_params[3*i+1] = honeycomb_positions[i, 1]
            best_params[3*i+2] = 0.0
        best_s = _outer_side_length_from_positions(honeycomb_positions, np.zeros(12))
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_params[3*i]
        inner_hex_data[i, 1] = best_params[3*i+1]
        inner_hex_data[i, 2] = best_params[3*i+2]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    s = _outer_side_length_from_positions(
        np.column_stack([best_params[0::3], best_params[1::3]]),
        best_params[2::3]
    )
    
    # Final validity check
    if not _check_validity(inner_hex_data, outer_hex_data, s):
        # Use a known valid fallback
        inner_hex_data = np.array([
            [0, 0, 0], [3, 0, 0], [1.5, sqrt3, 0], [-1.5, sqrt3, 0],
            [-3, 0, 0], [-1.5, -sqrt3, 0], [1.5, -sqrt3, 0],
            [4.5, sqrt3, 0], [3, 2*sqrt3, 0], [0, 2*sqrt3, 0],
            [-3, 2*sqrt3, 0], [-4.5, sqrt3, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        s = _outer_side_length_from_positions(honeycomb_positions, np.zeros(12))
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
