import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _hex_vertices_batch(positions, angle_deg):
    """Vertices for multiple hexagons at given positions, all with same rotation."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    base = np.column_stack([np.cos(angles), np.sin(angles)])  # (6,2)
    return positions[:, np.newaxis, :] + base[np.newaxis, :, :]


def _hex_support(angle_deg, theta):
    """Support function of unit hexagon: max dot product with direction theta."""
    a0 = math.radians(angle_deg)
    diff = (theta - a0) % (2 * math.pi)
    k = round(diff / (math.pi / 3.0))
    d1 = abs(diff - k * math.pi / 3.0)
    d2 = abs(diff - (k - 1) * math.pi / 3.0)
    d3 = abs(diff - (k + 1) * math.pi / 3.0)
    min_d = min(d1, d2, d3)
    return math.cos(min_d)


def _hex_distance_same_angle(d, angle_deg):
    """Minimum separation between two unit hexagons with same rotation angle.
    Returns positive if separated, negative if overlapping."""
    a = math.radians(angle_deg)
    dx, dy = d
    min_sep = float('inf')
    for k in range(6):
        theta = a + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(theta), math.sin(theta)
        proj = abs(dx * nx + dy * ny)
        s = math.sqrt(3) / 2.0
        sep = proj - 2 * s
        if sep < min_sep:
            min_sep = sep
    return min_sep


def _check_all_disjoint(positions, angle_deg, gap=0.001):
    """Check that all hexagons are pairwise disjoint with a small gap margin."""
    for i in range(N):
        for j in range(i + 1, N):
            d = (positions[j][0] - positions[i][0], positions[j][1] - positions[i][1])
            dist = _hex_distance_same_angle(d, angle_deg)
            if dist < -gap:
                return False
    return True


def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
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
        
        # (v - c).n <= R  =>  -n.c - R <= -v.n
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)
        
        # (c - v).n <= R  =>  n.c - R <= v.n
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_vals.append(v_dots)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    # R >= 0
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)
    
    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3 / 2)
            return cx, cy, side
    except Exception:
        pass
    return None


def _min_enclosing_hex_side_fast(verts, outer_angle_deg):
    """Fast centroid-based approximation."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    R = np.max(np.maximum(max_proj, -min_proj))
    if R < 0:
        R = 0
    side = R * 2.0 / _SQRT3
    return side


def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.0000001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
                 (-1, -1), (0, -2), (1, -2), (2, -1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 1:
        # Row-based: 3-4-4 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 3):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        # 4-3-4 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        # 2-4-2-3 pattern
        positions = []
        y1 = -3.0 * scale
        for k in range(0, 2):
            x = k * sqrt3 * scale - sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = -1.5 * scale
        for k in range(-1, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        y4 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        # 3-5-3 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 5:
        # Center + 6 ring + 4 more (different selection)
        positions = []
        positions.append((0, 0))
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        for ti, tj in ring1:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        ring2 = [(2, 0), (1, 1), (-1, 2), (-2, 0), (-1, -1), (0, -2)]
        for ti, tj in ring2[:4]:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 6:
        # 2-3-3-3 pattern
        positions = []
        y1 = -3.0 * scale
        for k in range(0, 2):
            x = k * sqrt3 * scale - sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        y4 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 7:
        # 4-4-3 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 8:
        # 5-6 pattern
        positions = []
        y1 = -0.75 * scale
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.75 * scale
        for k in range(-3, 3):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 9:
        # 6-5 pattern
        positions = []
        y1 = -0.75 * scale
        for k in range(-3, 3):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.75 * scale
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 10:
        # Diamond-ish
        positions = []
        y0 = -3.0 * scale
        positions.append((0, y0))
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        y4 = 3.0 * scale
        positions.append((0, y4))
        return positions[:N]
    elif layout_idx == 11:
        # 3-3-5 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        return positions[:N]
    else:
        # Fallback: compact grid
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        return coords_sorted[:N]


def _smooth_penalty(dist):
    """Smooth penalty: soft approximation of max(0, -dist)."""
    if dist >= 0:
        return 0.0
    d = -dist
    if d < 0.01:
        return d * d * 50.0
    elif d < 0.1:
        return d * d * 50.0 + (d - 0.01) * 0.5
    else:
        return d * 0.5 + 0.0045


def _objective_smooth(params, inner_angle):
    """Smooth objective: fast enclosing hex side + smooth penalty for overlaps."""
    n_pos = N * 2
    positions = np.array([(params[i], params[i + 1]) for i in range(0, n_pos, 2)])
    outer_angle = params[-1]
    
    all_vertices = _hex_vertices_batch(positions, inner_angle).reshape(-1, 2)
    side = _min_enclosing_hex_side_fast(all_vertices, outer_angle)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            di = positions[j] - positions[i]
            dist = _hex_distance_same_angle(di, inner_angle)
            if dist < 0:
                penalty += _smooth_penalty(dist)
    
    return side + penalty


def _objective_exact(params, inner_angle):
    """Exact objective: LP enclosing hex side + penalty for overlaps."""
    n_pos = N * 2
    positions = np.array([(params[i], params[i + 1]) for i in range(0, n_pos, 2)])
    outer_angle = params[-1]
    
    all_vertices = _hex_vertices_batch(positions, inner_angle).reshape(-1, 2)
    
    res = _min_enclosing_hex_side(all_vertices, outer_angle)
    if res is None:
        side = _min_enclosing_hex_side_fast(all_vertices, outer_angle)
    else:
        side = res[2]
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            di = positions[j] - positions[i]
            dist = _hex_distance_same_angle(di, inner_angle)
            if dist < 0:
                penalty += dist * dist * 100.0
    
    return side + penalty


def _optimize_layout(positions, inner_angle, outer_angles):
    """Optimize a layout by trying different outer angles and refining positions."""
    best_side = float('inf')
    best_pos = None
    best_outer = 0.0
    best_center = (0.0, 0.0)
    
    pos_arr = np.array(positions)
    all_vertices = _hex_vertices_batch(pos_arr, inner_angle).reshape(-1, 2)
    
    # First pass: try discrete outer angles with fast enclosing hex
    for outer_angle in outer_angles:
        side = _min_enclosing_hex_side_fast(all_vertices, outer_angle)
        if side < best_side:
            best_side = side
            best_pos = list(positions)
            best_outer = outer_angle
    
    # Second pass: refine with smooth Nelder-Mead
    if time.time() < _DEADLINE - 20.0:
        x0 = np.array([positions[i][0] for i in range(N)] +
                      [positions[i][1] for i in range(N)] + [best_outer])
        try:
            res = minimize(_objective_smooth, x0, args=(inner_angle,),
                          method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_pos = [(res.x[i], res.x[i + 1]) for i in range(0, N * 2, 2)]
                best_outer = res.x[-1]
        except Exception:
            pass
    
    # Third pass: high-precision polish with exact LP-based objective
    if time.time() < _DEADLINE - 10.0 and best_pos is not None:
        x0 = np.array([best_pos[i][0] for i in range(N)] +
                      [best_pos[i][1] for i in range(N)] + [best_outer])
        try:
            res2 = minimize(_objective_exact, x0, args=(inner_angle,),
                           method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res2.fun < best_side:
                best_side = res2.fun
                best_pos = [(res2.x[i], res2.x[i + 1]) for i in range(0, N * 2, 2)]
                best_outer = res2.x[-1]
        except Exception:
            pass
    
    # Compute exact center via LP
    if best_pos is not None:
        pos_arr2 = np.array(best_pos)
        all_vertices2 = _hex_vertices_batch(pos_arr2, inner_angle).reshape(-1, 2)
        res_lp = _min_enclosing_hex_side(all_vertices2, best_outer)
        if res_lp is not None:
            best_center = (res_lp[0], res_lp[1])
            best_side = res_lp[2]
    
    return best_side, best_pos, best_outer, best_center


def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    # Outer angles to try (finer grid)
    outer_angles = list(np.arange(0, 30.0, 0.5))
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 12.0:
            break
        
        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 12.0:
                break
            
            refined_side, refined_pos, refined_angle, refined_center = _optimize_layout(
                positions, inner_angle, outer_angles)
            
            if refined_side < best_side:
                best_side = refined_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in refined_pos]
                best_center = refined_center
                best_outer_angle = refined_angle
    
    # Final: compute exact enclosing hex side via LP for the best configuration
    if best_inner is not None:
        pos_arr = np.array([[p[0], p[1]] for p in best_inner])
        inner_angle = best_inner[0][2]
        all_vertices = _hex_vertices_batch(pos_arr, inner_angle).reshape(-1, 2)
        
        # Fine-tune outer angle around the best found
        best_final_side = float('inf')
        best_final_center = (0.0, 0.0)
        best_final_angle = best_outer_angle
        
        for da in np.arange(-1.0, 1.01, 0.05):
            test_angle = best_outer_angle + da
            res_lp = _min_enclosing_hex_side(all_vertices, test_angle)
            if res_lp is not None:
                cx, cy, side_test = res_lp
                if side_test < best_final_side:
                    best_final_side = side_test
                    best_final_center = (cx, cy)
                    best_final_angle = test_angle
        
        best_center = best_final_center
        best_side = best_final_side
        best_outer_angle = best_final_angle
        
        # Verify disjointness; if not disjoint, add small gap
        if not _check_all_disjoint(pos_arr, inner_angle, gap=0.0005):
            scale = 1.002
            pos_arr = pos_arr * scale
            best_inner = [[pos_arr[i][0], pos_arr[i][1], inner_angle] for i in range(N)]
            all_vertices = _hex_vertices_batch(pos_arr, inner_angle).reshape(-1, 2)
            
            res_lp = _min_enclosing_hex_side(all_vertices, best_outer_angle)
            if res_lp is not None:
                best_center = (res_lp[0], res_lp[1])
                best_side = res_lp[2]
    
    # High-precision polish using L-BFGS-B on the exact objective
    if time.time() < _DEADLINE - 5.0 and best_inner is not None:
        inner_angle = best_inner[0][2]
        x0 = np.array([best_inner[i][0] for i in range(N)] +
                      [best_inner[i][1] for i in range(N)] + [best_outer_angle])
        
        try:
            res_polish = minimize(_objective_exact, x0, args=(inner_angle,),
                                  method='L-BFGS-B',
                                  options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res_polish.success and res_polish.fun < best_side:
                new_pos = [(res_polish.x[i], res_polish.x[i + 1]) for i in range(0, N * 2, 2)]
                new_angle = res_polish.x[-1]
                
                new_pos_arr = np.array(new_pos)
                if _check_all_disjoint(new_pos_arr, inner_angle, gap=0.0):
                    best_inner = [[p[0], p[1], inner_angle] for p in new_pos]
                    best_outer_angle = new_angle
                    
                    pos_arr = np.array([[p[0], p[1]] for p in best_inner])
                    all_vertices = _hex_vertices_batch(pos_arr, inner_angle).reshape(-1, 2)
                    res_lp = _min_enclosing_hex_side(all_vertices, best_outer_angle)
                    if res_lp is not None:
                        best_center = (res_lp[0], res_lp[1])
                        best_side = res_lp[2]
        except Exception:
            pass
    
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
