import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    return np.array([(cx + math.cos(a0 + math.pi*k/3), cy + math.sin(a0 + math.pi*k/3)) for k in range(6)])

def _hex_vertices_fast(cx, cy, angle_rad):
    """Fast vectorized hex vertices."""
    angles = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    normals = np.array([(math.cos(math.radians(outer_angle_deg + 30 + 60*k)),
                         math.sin(math.radians(outer_angle_deg + 30 + 60*k))) for k in range(6)])
    
    M = len(verts)
    A = []
    b = []
    for i in range(M):
        vx, vy = verts[i]
        for nx, ny in normals:
            vd = vx*nx + vy*ny
            A.append([-nx, -ny, -1.0])
            b.append(-vd)
            A.append([nx, ny, -1.0])
            b.append(vd)
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3)/2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _hex_pair_distance_penalty(v1, v2, w1, w2):
    """
    Compute a smooth penalty for two hexagons overlapping.
    v1, v2: (6,2) vertex arrays; w1, w2: (6,2) vertex arrays (rotated normals/edges)
    Uses minimum edge-edge distance approximation.
    """
    # For each pair of vertices from hex1 and hex2, compute distance
    # Also check if any vertex of one is inside the other
    # Simple approach: min distance between any vertex of hex1 and any edge of hex2, and vice versa
    
    def point_to_segment_dist(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)
    
    min_dist = float('inf')
    # Check vertices of hex1 against edges of hex2
    for i in range(6):
        for j in range(6):
            d = point_to_segment_dist(v1[i], v2[j], v2[(j+1)%6])
            if d < min_dist:
                min_dist = d
    # Check vertices of hex2 against edges of hex1
    for i in range(6):
        for j in range(6):
            d = point_to_segment_dist(v2[i], v1[j], v1[(j+1)%6])
            if d < min_dist:
                min_dist = d
    
    # Penalty: if min_dist < 0 (overlap), large penalty; else small
    # Use soft penalty: penalty = max(0, 0.01 - min_dist)^2
    overlap = max(0.0, 0.005 - min_dist)
    return overlap * overlap * 100.0

def _nonlinear_packing_optimize(positions, angles, outer_angle_deg, max_iters=500):
    """
    Nonlinear optimization to minimize outer hexagon side length.
    Variables: for each of N hexagons, (x, y, angle_rad) = 3N variables.
    Objective: outer side length + penalty for overlaps.
    """
    N_local = len(positions)
    n_vars = 3 * N_local
    
    def _get_all_vertices(x):
        """Get all vertices from current variables."""
        verts_list = []
        for i in range(N_local):
            cx = x[3*i]
            cy = x[3*i+1]
            ang = x[3*i+2]
            verts_list.append(_hex_vertices_fast(cx, cy, ang))
        return verts_list
    
    def _objective(x):
        verts_list = _get_all_vertices(x)
        
        # Compute enclosing hexagon side
        all_verts = np.vstack(verts_list)
        res = _min_enclosing_hex_side(all_verts, outer_angle_deg)
        if res is None:
            return 1e6
        side = res[2]
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N_local):
            for j in range(i+1, N_local):
                penalty += _hex_pair_distance_penalty(verts_list[i], verts_list[j], None, None)
        
        return side + penalty
    
    # Initial guess
    x0 = np.zeros(n_vars)
    for i in range(N_local):
        x0[3*i] = positions[i][0]
        x0[3*i+1] = positions[i][1]
        x0[3*i+2] = math.radians(angles[i])
    
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.success:
            return result.x, result.fun
        else:
            # Still try to use the result if it improved
            if result.fun < _objective(x0):
                return result.x, result.fun
    except Exception:
        pass
    
    return x0, _objective(x0)

def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = math.sqrt(3)
    scale = 1.0000001
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    coords = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i*u[0] + j*v[0]
            y = i*u[1] + j*v[1]
            coords.append((x, y, i, j))
    
    if layout_idx == 0:
        target = [(0,0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x,y))
        return positions[:N]
    elif layout_idx == 1:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return [(x,y) for x,y,i,j in coords_sorted[:N]]

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    layouts = [_build_positions(i) for i in range(4)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 10.0:
                break
            
            all_vertices = []
            for px, py in positions:
                all_vertices.extend(_hex_vertices(px, py, inner_angle))
            all_vertices = np.array(all_vertices)
            
            best_coarse_side = float('inf')
            best_coarse_angle = 0.0
            best_coarse_center = (0.0, 0.0)
            
            for outer_angle_deg in np.arange(0, 30.1, 1.0):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_angle = outer_angle_deg
                        best_coarse_center = (cx, cy)
            
            if best_coarse_angle > 0:
                for outer_angle_deg in np.arange(max(0, best_coarse_angle - 1.0),
                                                 min(30.1, best_coarse_angle + 1.0), 0.05):
                    if time.time() > _DEADLINE - 10.0:
                        break
                    res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < best_coarse_side:
                            best_coarse_side = side
                            best_coarse_angle = outer_angle_deg
                            best_coarse_center = (cx, cy)
            
            # Nonlinear polish: try to improve by allowing small adjustments
            if best_coarse_side < best_side and time.time() < _DEADLINE - 20.0:
                try:
                    angles_list = [inner_angle] * N
                    x_opt, obj_val = _nonlinear_packing_optimize(
                        positions, angles_list, best_coarse_angle, max_iters=200
                    )
                    # Extract optimized positions
                    opt_positions = []
                    opt_angles = []
                    for i in range(N):
                        px = x_opt[3*i]
                        py = x_opt[3*i+1]
                        pa = math.degrees(x_opt[3*i+2])
                        opt_positions.append((px, py))
                        opt_angles.append(pa)
                    
                    # Verify with enclosing hex
                    all_v_opt = []
                    for i in range(N):
                        all_v_opt.extend(_hex_vertices(opt_positions[i][0], opt_positions[i][1], opt_angles[i]))
                    all_v_opt = np.array(all_v_opt)
                    
                    res_opt = _min_enclosing_hex_side(all_v_opt, best_coarse_angle)
                    if res_opt:
                        cx_o, cy_o, side_o = res_opt
                        # Check for overlaps
                        has_overlap = False
                        for i in range(N):
                            for j in range(i+1, N):
                                d = _hex_pair_distance_penalty(
                                    _hex_vertices_fast(opt_positions[i][0], opt_positions[i][1], math.radians(opt_angles[i])),
                                    _hex_vertices_fast(opt_positions[j][0], opt_positions[j][1], math.radians(opt_angles[j])),
                                    None, None)
                                if d > 1e-6:
                                    has_overlap = True
                                    break
                            if has_overlap:
                                break
                        
                        if not has_overlap and side_o < best_coarse_side:
                            best_coarse_side = side_o
                            best_coarse_center = (cx_o, cy_o)
                            positions = opt_positions
                            inner_angle = opt_angles[0]  # Store for return
                except Exception:
                    pass
            
            if best_coarse_side < best_side:
                best_side = best_coarse_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_coarse_center
                best_outer_angle = best_coarse_angle
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
