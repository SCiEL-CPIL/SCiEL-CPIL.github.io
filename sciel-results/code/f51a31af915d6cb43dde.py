import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """Vectorized LP for minimum enclosing regular hexagon."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    v_dot_n = verts @ normals.T  # (M, 6)
    
    A_rows = []
    b_vals = []
    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_vals.append(v_dots)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _hex_distance_fast(p1, a1, p2, a2):
    """Fast minimum distance between two unit hexagons (radians angles)."""
    d = np.linalg.norm(p1 - p2)
    if d < 1e-10:
        return 0.0
    dir_vec = (p2 - p1) / d
    theta = math.atan2(dir_vec[1], dir_vec[0])
    
    vals1 = np.cos(a1 + np.arange(6) * math.pi / 3.0 - theta)
    vals2 = np.cos(a2 + np.arange(6) * math.pi / 3.0 - (theta + math.pi))
    h1 = np.max(vals1)
    h2 = np.max(vals2)
    return d - h1 - h2

def _all_pairwise_dists(positions, angles):
    """Compute all pairwise distances between hexagons. positions: list of (x,y), angles: list of radians."""
    n = len(positions)
    pos_arr = np.array(positions)
    ang_arr = np.array(angles)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = _hex_distance_fast(pos_arr[i], ang_arr[i], pos_arr[j], ang_arr[j])
            dists.append(d)
    return np.array(dists)

def _softmin(vals, beta=100.0):
    """Smooth minimum via log-sum-exp."""
    m = np.min(vals)
    shifted = vals - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return m - log_sum / beta

def _build_positions(layout_idx):
    scale = 1.0000001
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
        target = [(0, 0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 1:
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k * _SQRT3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 3):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-2, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        positions = []
        y1 = -3.0 * scale
        for k in range(0, 2):
            x = k * _SQRT3 * scale - _SQRT3 / 2 * scale
            positions.append((x, y1))
        y2 = -1.5 * scale
        for k in range(-1, 3):
            x = k * _SQRT3 * scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y3))
        y4 = 1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k * _SQRT3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 5:
        positions = [(0, 0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        for ti, tj in ring1:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        ring2 = [(2,0),(1,1),(-1,2),(-2,0),(-1,-1),(0,-2)]
        for ti, tj in ring2[:4]:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 6:
        positions = []
        y1 = -3.0 * scale
        for k in range(0, 2):
            x = k * _SQRT3 * scale - _SQRT3 / 2 * scale
            positions.append((x, y1))
        y2 = -1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 2):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y3))
        y4 = 1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 7:
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 8:
        positions = []
        y1 = -0.75 * scale
        for k in range(-2, 3):
            x = k * _SQRT3 * scale
            positions.append((x, y1))
        y2 = 0.75 * scale
        for k in range(-3, 3):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 9:
        positions = []
        y1 = -0.75 * scale
        for k in range(-3, 3):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.75 * scale
        for k in range(-2, 3):
            x = k * _SQRT3 * scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 10:
        positions = []
        y0 = -3.0 * scale
        positions.append((0, y0))
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k * _SQRT3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y3))
        y4 = 3.0 * scale
        positions.append((0, y4))
        return positions[:N]
    elif layout_idx == 11:
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-2, 3):
            x = k * _SQRT3 * scale
            positions.append((x, y3))
        return positions[:N]
    else:
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]

def _objective(params, base_positions, outer_angle_deg, beta=80.0):
    """
    Objective: outer hexagon side + smooth barrier for overlaps.
    params: [x0,y0,a0, x1,y1,a1, ...] (3N values)
    base_positions: not used directly, params are absolute.
    """
    n = len(base_positions)
    positions = []
    angles_rad = []
    for i in range(n):
        positions.append((params[3*i], params[3*i+1]))
        angles_rad.append(params[3*i+2])
    
    # Compute pairwise distances
    dists = _all_pairwise_dists(positions, angles_rad)
    if len(dists) == 0:
        dists = np.array([1.0])
    
    sm = _softmin(dists, beta)
    barrier = max(0.0, -sm) ** 2 * 50.0
    
    # Compute enclosing hexagon - vectorized vertex computation
    pos_arr = np.array(positions)
    ang_deg = np.degrees(np.array(angles_rad))
    base_angles = np.radians(ang_deg)[:, None] + np.arange(6) * math.pi / 3.0
    cos_a = np.cos(base_angles)
    sin_a = np.sin(base_angles)
    all_vertices = np.column_stack([
        pos_arr[:, 0:1].repeat(6, axis=1) + cos_a,
        pos_arr[:, 1:2].repeat(6, axis=1) + sin_a
    ])
    
    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    return side + barrier

def _objective_no_barrier(params, base_positions, outer_angle_deg):
    """Objective without barrier (for final polish when no overlaps)."""
    n = len(base_positions)
    positions = []
    angles = []
    for i in range(n):
        positions.append((params[3*i], params[3*i+1]))
        angles.append(params[3*i+2])
    
    dists = _all_pairwise_dists(positions, angles)
    if len(dists) > 0 and np.min(dists) < -1e-6:
        return 1000.0
    
    all_vertices = []
    for i in range(n):
        verts = _hex_vertices(positions[i][0], positions[i][1], math.degrees(angles[i]))
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    return side

def _apply_affine(positions, params):
    """
    Apply affine transformation to base positions.
    params: [sx, sy, shear, rot_rad, tx, ty]
    Returns transformed positions.
    """
    sx, sy, shear, rot_rad, tx, ty = params
    c, s = math.cos(rot_rad), math.sin(rot_rad)
    # Affine matrix: R * S where S = [[sx, shear],[0, sy]], R = [[c,-s],[s,c]]
    # Combined: [[c*sx, -s*sx + c*shear*sy], [s*sx, c*sx + s*shear*sy]]
    # Actually let's do: first scale+shear, then rotate, then translate
    # P' = R * (S * P) + t
    M = np.array([[sx, shear], [0.0, sy]])
    R = np.array([[c, -s], [s, c]])
    A = R @ M  # 2x2
    
    pos_arr = np.array(positions)
    transformed = (A @ pos_arr.T).T + np.array([tx, ty])
    return [tuple(p) for p in transformed]

def construct():
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_params = None
    
    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    # Phase 0: Evaluate all layouts with grid search over outer angle
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 15.0:
            break
        
        inner_angle = 30.0
        
        all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
        
        best_local_side = float('inf')
        best_local_center = (0.0, 0.0)
        best_local_angle = 0.0
        
        # Coarse grid
        for outer_angle_deg in np.arange(0, 30.0, 1.0):
            if time.time() > _DEADLINE - 15.0:
                break
            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_local_side:
                    best_local_side = side
                    best_local_center = (cx, cy)
                    best_local_angle = outer_angle_deg
        
        # Fine grid around best
        if best_local_side < float('inf'):
            for outer_angle_deg in np.arange(max(0, best_local_angle - 1.0), min(30.0, best_local_angle + 1.0), 0.05):
                if time.time() > _DEADLINE - 15.0:
                    break
                res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_local_side:
                        best_local_side = side
                        best_local_center = (cx, cy)
                        best_local_angle = outer_angle_deg
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_center = best_local_center
            best_outer_angle = best_local_angle
            best_params = []
            for pos in positions:
                best_params.append(pos[0])
                best_params.append(pos[1])
                best_params.append(math.radians(inner_angle))
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Phase 1: Local optimization with smooth barrier (L-BFGS-B)
    if time.time() < _DEADLINE - 30.0:
        x0 = best_params[:]
        
        # Try a few outer angles around best
        outer_angles_to_try = list(np.arange(max(0, best_outer_angle - 3), min(30, best_outer_angle + 3), 1.0))
        
        for outer_angle_deg in outer_angles_to_try:
            if time.time() > _DEADLINE - 30.0:
                break
            
            try:
                res = minimize(_objective, x0, args=([], outer_angle_deg, 100.0),
                              method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if res.fun < best_side and res.fun < 100:
                    new_positions = []
                    new_angles = []
                    for i in range(N):
                        new_positions.append((res.x[3*i], res.x[3*i+1]))
                        new_angles.append(math.degrees(res.x[3*i+2]))
                    
                    dists = _all_pairwise_dists(new_positions, [math.radians(a) for a in new_angles])
                    if len(dists) > 0 and np.min(dists) < -1e-4:
                        continue
                    
                    pos_arr = np.array(new_positions)
                    ang_deg_arr = np.array(new_angles)
                    base_angles = np.radians(ang_deg_arr)[:, None] + np.arange(6) * math.pi / 3.0
                    cos_a = np.cos(base_angles)
                    sin_a = np.sin(base_angles)
                    all_vertices = np.column_stack([
                        pos_arr[:, 0:1].repeat(6, axis=1) + cos_a,
                        pos_arr[:, 1:2].repeat(6, axis=1) + sin_a
                    ])
                    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                    
                    if side < best_side:
                        best_side = side
                        best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                        best_center = (cx, cy)
                        best_outer_angle = outer_angle_deg
                        best_params = res.x[:]
            except Exception:
                pass
    
    # Phase 2: Affine shape morphing - apply affine transforms to base layout
    if time.time() < _DEADLINE - 20.0:
        base_positions = [(p[0], p[1]) for p in best_inner]
        
        # Reduced search: fewer initial points, use COBYLA which is faster for small dims
        init_configs = [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.97, 1.03, 0.0, 0.0, 0.0, 0.0],
            [1.03, 0.97, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.05, 0.0, 0.0, 0.0],
            [1.0, 1.0, -0.05, 0.0, 0.0, 0.0],
        ]
        
        for affine_params0 in init_configs:
            if time.time() > _DEADLINE - 20.0:
                break
            
            def _affine_objective(ap):
                if time.time() > _DEADLINE - 18.0:
                    return 1000.0
                transformed = _apply_affine(base_positions, ap)
                angles_rad = np.full(N, math.radians(30.0))
                dists = _all_pairwise_dists(transformed, angles_rad)
                if len(dists) > 0 and np.min(dists) < -0.01:
                    return 100.0 + abs(np.min(dists)) * 100
                
                pos_arr = np.array(transformed)
                base_angles = np.radians(30.0) + np.arange(6) * math.pi / 3.0
                cos_a = np.cos(base_angles)
                sin_a = np.sin(base_angles)
                all_vertices = np.column_stack([
                    pos_arr[:, 0:1].repeat(6, axis=1) + np.tile(cos_a, (N, 1)),
                    pos_arr[:, 1:2].repeat(6, axis=1) + np.tile(sin_a, (N, 1))
                ])
                cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, best_outer_angle)
                if len(dists) > 0:
                    sm = _softmin(dists, 100.0)
                    barrier = max(0.0, -sm) ** 2 * 50.0
                else:
                    barrier = 0.0
                return side + barrier
            
            try:
                res = minimize(_affine_objective, np.array(affine_params0),
                              method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
                if res.fun < best_side and res.fun < 100:
                    transformed = _apply_affine(base_positions, res.x)
                    angles_rad = np.full(N, math.radians(30.0))
                    dists = _all_pairwise_dists(transformed, angles_rad)
                    if len(dists) > 0 and np.min(dists) < -1e-4:
                        continue
                    pos_arr = np.array(transformed)
                    base_angles = np.radians(30.0) + np.arange(6) * math.pi / 3.0
                    cos_a = np.cos(base_angles)
                    sin_a = np.sin(base_angles)
                    all_vertices = np.column_stack([
                        pos_arr[:, 0:1].repeat(6, axis=1) + np.tile(cos_a, (N, 1)),
                        pos_arr[:, 1:2].repeat(6, axis=1) + np.tile(sin_a, (N, 1))
                    ])
                    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, best_outer_angle)
                    if side < best_side:
                        best_side = side
                        best_inner = [[px, py, 30.0] for px, py in transformed]
                        best_center = (cx, cy)
                        best_params = []
                        for px, py in transformed:
                            best_params.append(px)
                            best_params.append(py)
                            best_params.append(math.radians(30.0))
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish with outer angle co-optimization
    if time.time() < _DEADLINE - 5.0 and best_params is not None:
        x0 = best_params[:]
        
        for polish_iter in range(4):
            if time.time() > _DEADLINE - 5.0:
                break
            
            try:
                res = minimize(_objective_no_barrier, x0, args=([], best_outer_angle),
                              method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
                
                if res.fun < best_side and res.fun < 100:
                    new_positions = []
                    new_angles = []
                    for i in range(N):
                        new_positions.append((res.x[3*i], res.x[3*i+1]))
                        new_angles.append(math.degrees(res.x[3*i+2]))
                    
                    dists = _all_pairwise_dists(new_positions, [math.radians(a) for a in new_angles])
                    if len(dists) > 0 and np.min(dists) < -1e-5:
                        break
                    
                    pos_arr = np.array(new_positions)
                    ang_deg_arr = np.array(new_angles)
                    base_angles = np.radians(ang_deg_arr)[:, None] + np.arange(6) * math.pi / 3.0
                    cos_a = np.cos(base_angles)
                    sin_a = np.sin(base_angles)
                    all_vertices = np.column_stack([
                        pos_arr[:, 0:1].repeat(6, axis=1) + cos_a,
                        pos_arr[:, 1:2].repeat(6, axis=1) + sin_a
                    ])
                    
                    # Also try nearby outer angles for this configuration
                    best_ao = best_outer_angle
                    best_ao_side = side
                    for ao in np.arange(max(0, best_outer_angle - 2), min(30, best_outer_angle + 2), 0.1):
                        if time.time() > _DEADLINE - 4.0:
                            break
                        cx2, cy2, side2 = _min_enclosing_hex_side_lp(all_vertices, ao)
                        if side2 < best_ao_side:
                            best_ao_side = side2
                            best_ao = ao
                            cx, cy = cx2, cy2
                    
                    if best_ao_side < best_side:
                        best_side = best_ao_side
                        best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                        best_center = (cx, cy)
                        best_outer_angle = best_ao
                        best_params = res.x[:]
                    else:
                        break
                else:
                    break
            except Exception:
                break
    
    # Final verification and safety
    if best_inner is None or best_side >= float('inf'):
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Ensure exactly N entries
    best_inner = best_inner[:N]
    while len(best_inner) < N:
        best_inner.append([0.0, 0.0, 0.0])
    
    # Final sanity check: verify no overlaps
    final_positions = [(p[0], p[1]) for p in best_inner]
    final_angles = [math.radians(p[2]) for p in best_inner]
    final_dists = _all_pairwise_dists(final_positions, final_angles)
    if len(final_dists) > 0 and np.min(final_dists) < -1e-4:
        # Overlap detected, fall back to safe layout
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
