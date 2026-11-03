import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

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
    v_dot_n = verts @ normals.T
    
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

def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    scale = 1.000001
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)
    
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
        # Row-based: 3-4-4
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
        # 4-3-4 shifted
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
        # 5-3-3 arrangement
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 3):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 4:
        # Alternative compact: center + 6 + different 4 from ring 2
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        # Pick 4 that form a more symmetric pattern
        ring2_picks = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        target.extend(ring2_picks)
        
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    else:
        # Fallback: sorted by distance
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        return coords_sorted[:N]

def _enclosing_side_for_positions(pos_flat, n_hex, inner_angle, outer_angle_deg):
    """
    Compute enclosing hexagon side length for flat position array.
    pos_flat: array of shape (2*n_hex,) with [x0,y0,x1,y1,...]
    Returns side length or a large penalty.
    """
    if time.time() > _DEADLINE - 8.0:
        return 1e10
    
    pos = pos_flat.reshape(n_hex, 2)
    
    # Quick overlap check using center distances
    min_center_dist = _SQRT3 * 0.9999  # minimum center distance for non-overlap
    for i in range(n_hex):
        for j in range(i + 1, n_hex):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_center_dist - 0.01:
                return 1e10 + (min_center_dist - d) * 100
    
    all_vertices = []
    for px, py in pos:
        all_vertices.extend(_hex_vertices(px, py, inner_angle))
    all_vertices = np.array(all_vertices)
    
    res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
    if res:
        _, _, side = res
        return side
    return 1e10

def _optimize_positions(initial_positions, inner_angle, outer_angle_deg, max_iter=100):
    """
    Optimize positions to minimize enclosing hexagon side length.
    Uses scipy minimize with L-BFGS-B.
    """
    n = len(initial_positions)
    x0 = np.array(initial_positions, dtype=float).flatten()
    
    def objective(x):
        return _enclosing_side_for_positions(x, n, inner_angle, outer_angle_deg)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'maxls': 20})
        if result.success or result.fun < 1e9:
            return result.x.reshape(n, 2)
    except Exception:
        pass
    
    return np.array(initial_positions, dtype=float)

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    # Generate layouts
    layouts = [_build_positions(i) for i in range(5)]
    
    # Also generate rotated versions of the first layout
    base_layout = layouts[0]
    for rot_deg in [15.0, 45.0]:
        rot_rad = math.radians(rot_deg)
        c, s = math.cos(rot_rad), math.sin(rot_rad)
        rotated = []
        for x, y in base_layout:
            rx = c * x - s * y
            ry = s * x + c * y
            rotated.append((rx, ry))
        layouts.append(rotated)
    
    # Try different inner angles
    inner_angles = [0.0, 30.0, 15.0, 45.0]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 15.0:
            break
        
        for inner_angle in inner_angles:
            if time.time() > _DEADLINE - 12.0:
                break
            
            # First, find best outer angle with coarse search
            all_vertices = []
            for px, py in positions:
                all_vertices.extend(_hex_vertices(px, py, inner_angle))
            all_vertices = np.array(all_vertices)
            
            best_coarse_side = float('inf')
            best_coarse_angle = 0.0
            best_coarse_center = (0.0, 0.0)
            
            # Coarse search: 0 to 30 degrees in 1-degree steps
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
            
            if best_coarse_side >= best_side:
                continue
            
            # Fine search around best coarse angle
            fine_start = max(0, best_coarse_angle - 1.0)
            fine_end = min(30.1, best_coarse_angle + 1.0)
            for outer_angle_deg in np.arange(fine_start, fine_end, 0.05):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_angle = outer_angle_deg
                        best_coarse_center = (cx, cy)
            
            # Now optimize positions with the best outer angle
            if time.time() < _DEADLINE - 20.0:
                optimized = _optimize_positions(positions, inner_angle, best_coarse_angle, max_iter=50)
                
                # Evaluate optimized positions
                all_vertices_opt = []
                for px, py in optimized:
                    all_vertices_opt.extend(_hex_vertices(px, py, inner_angle))
                all_vertices_opt = np.array(all_vertices_opt)
                
                # Re-find best outer angle for optimized positions
                best_opt_side = float('inf')
                best_opt_angle = best_coarse_angle
                best_opt_center = best_coarse_center
                
                for outer_angle_deg in np.arange(max(0, best_coarse_angle - 0.5),
                                                 min(30.1, best_coarse_angle + 0.5), 0.02):
                    if time.time() > _DEADLINE - 10.0:
                        break
                    res = _min_enclosing_hex_side(all_vertices_opt, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < best_opt_side:
                            best_opt_side = side
                            best_opt_angle = outer_angle_deg
                            best_opt_center = (cx, cy)
                
                if best_opt_side < best_side:
                    best_side = best_opt_side
                    best_inner = [[pos[0], pos[1], inner_angle] for pos in optimized]
                    best_center = best_opt_center
                    best_outer_angle = best_opt_angle
                    best_inner_angle = inner_angle
            
            # Also keep the non-optimized result if it's better
            if best_coarse_side < best_side:
                best_side = best_coarse_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_coarse_center
                best_outer_angle = best_coarse_angle
                best_inner_angle = inner_angle
    
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
