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

def _min_enclosing_hex_global(verts):
    """
    Find minimum enclosing regular hexagon over all orientations and centers.
    Returns (cx, cy, side, outer_angle_deg) or None.
    Uses a two-stage approach: coarse grid over angle, then refine.
    """
    best_side = float('inf')
    best_params = None
    
    # Coarse search over outer angle
    for outer_angle_deg in np.arange(0, 30.1, 0.5):
        if time.time() > _DEADLINE - 15.0:
            break
        res = _min_enclosing_hex_side(verts, outer_angle_deg)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_params = (cx, cy, side, outer_angle_deg)
    
    # Fine search around best angle
    if best_params:
        best_angle = best_params[3]
        for outer_angle_deg in np.arange(max(0, best_angle - 0.5),
                                         min(30.1, best_angle + 0.5), 0.02):
            if time.time() > _DEADLINE - 15.0:
                break
            res = _min_enclosing_hex_side(verts, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_params = (cx, cy, side, outer_angle_deg)
    
    return best_params

def _build_hexagonal_positions():
    """Build 11 unit hexagons in a compact hexagonal arrangement."""
    scale = 1.0000001
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)
    
    positions = []
    # Center
    positions.append((0, 0))
    # Ring 1: 6 hexagons
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for ti, tj in ring1:
        x = ti*u[0] + tj*v[0]
        y = ti*u[1] + tj*v[1]
        positions.append((x, y))
    # Ring 2: pick 4 closest to center
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)]
    ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
    for ti, tj in ring2_sorted[:4]:
        x = ti*u[0] + tj*v[0]
        y = ti*u[1] + tj*v[1]
        positions.append((x, y))
    
    return positions[:N]

def _build_row_positions():
    """Build 11 unit hexagons in rows."""
    scale = 1.0000001
    positions = []
    # Row at y=0: 5 hexagons
    for k in range(-2, 3):
        positions.append((k*_SQRT3*scale, 0.0))
    # Row at y=1.5: 3 hexagons (offset)
    for k in range(-1, 2):
        positions.append((k*_SQRT3*scale + _SQRT3/2*scale, 1.5*scale))
    # Row at y=-1.5: 3 hexagons (offset)
    for k in range(-1, 2):
        positions.append((k*_SQRT3*scale + _SQRT3/2*scale, -1.5*scale))
    return positions[:N]

def _build_compact_positions():
    """Build 11 unit hexagons in a more compact arrangement."""
    scale = 1.0000001
    positions = []
    # Try a 4-3-4 arrangement
    # Bottom row: 4 hexagons
    for k in range(-2, 2):
        positions.append((k*_SQRT3*scale, -1.5*scale))
    # Middle row: 3 hexagons (offset)
    for k in range(-1, 2):
        positions.append((k*_SQRT3*scale + _SQRT3/2*scale, 0.0))
    # Top row: 4 hexagons
    for k in range(-2, 2):
        positions.append((k*_SQRT3*scale, 1.5*scale))
    return positions[:N]

def _optimize_positions_continuous(positions, inner_angle):
    """
    Optimize positions and inner angle continuously to minimize enclosing hexagon side.
    Uses scipy.optimize.minimize with a penalty-based objective.
    """
    # Flatten positions to 1D array
    init_x = np.array([p[0] for p in positions] + [p[1] for p in positions] + [inner_angle])
    
    def objective(params):
        if time.time() > _DEADLINE - 15.0:
            return 1e10
        try:
            pos_x = params[:N]
            pos_y = params[N:2*N]
            angle = params[2*N]
            
            all_vertices = []
            for i in range(N):
                verts = _hex_vertices(pos_x[i], pos_y[i], angle)
                all_vertices.extend(verts)
            all_vertices = np.array(all_vertices)
            
            # Try several outer angles
            best_side = float('inf')
            for outer_angle in [0.0, 15.0, 30.0]:
                res = _min_enclosing_hex_side(all_vertices, outer_angle)
                if res:
                    _, _, side = res
                    if side < best_side:
                        best_side = side
            
            return best_side
        except Exception:
            return 1e10
    
    try:
        res = minimize(objective, init_x, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.success:
            pos_x = res.x[:N]
            pos_y = res.x[N:2*N]
            angle = res.x[2*N]
            new_positions = list(zip(pos_x, pos_y))
            return new_positions, angle, res.fun
    except Exception:
        pass
    
    return positions, inner_angle, float('inf')

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Generate initial layouts
    layouts = [
        _build_hexagonal_positions(),
        _build_row_positions(),
        _build_compact_positions()
    ]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        for inner_angle in [0.0, 30.0, 15.0]:
            if time.time() > _DEADLINE - 10.0:
                break
            
            all_vertices = []
            for px, py in positions:
                all_vertices.extend(_hex_vertices(px, py, inner_angle))
            all_vertices = np.array(all_vertices)
            
            # Coarse search over outer angle
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
            
            # Fine search
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
            
            if best_coarse_side < best_side:
                best_side = best_coarse_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_coarse_center
                best_outer_angle = best_coarse_angle
    
    # Try continuous optimization on best layout
    if best_inner is not None and time.time() < _DEADLINE - 20.0:
        positions = [(p[0], p[1]) for p in best_inner]
        inner_angle = best_inner[0][2]
        
        opt_positions, opt_angle, opt_side = _optimize_positions_continuous(
            positions, inner_angle)
        
        if opt_positions is not None and opt_side < best_side:
            # Compute enclosing hexagon for optimized positions
            all_vertices = []
            for px, py in opt_positions:
                all_vertices.extend(_hex_vertices(px, py, opt_angle))
            all_vertices = np.array(all_vertices)
            
            best_opt_side = float('inf')
            best_opt_angle = 0.0
            best_opt_center = (0.0, 0.0)
            
            for outer_angle_deg in np.arange(max(0, best_outer_angle - 2.0),
                                             min(30.1, best_outer_angle + 2.0), 0.1):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_opt_side:
                        best_opt_side = side
                        best_opt_angle = outer_angle_deg
                        best_opt_center = (cx, cy)
            
            if best_opt_side < best_side:
                best_side = best_opt_side
                best_inner = [[pos[0], pos[1], opt_angle] for pos in opt_positions]
                best_center = best_opt_center
                best_outer_angle = best_opt_angle
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
