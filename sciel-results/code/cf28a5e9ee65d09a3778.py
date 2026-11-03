import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0 # 5s buffer

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _find_min_enclosing_hex_side(vertices, outer_angle_deg):
    """
    Find the minimum side length of a regular hexagon with given angle
    that contains all given vertices.
    Returns (center_x, center_y, side_length) or None.
    """
    # Normals for the 3 unique directions of the hexagon sides
    # The sides are perpendicular to these normals.
    # A regular hexagon has normals at 30, 90, 150, 210, 270, 330 relative to its orientation.
    # Due to symmetry, we only need 3 unique normals for the constraints |n . (v-c)| <= h
    base_angle = math.radians(outer_angle_deg + 30)
    normals = []
    for k in range(3):
        a = base_angle + math.pi * k / 3
        normals.append((math.cos(a), math.sin(a)))
    
    A = []
    b = []
    
    # Variables: [cx, cy, h]
    # Constraints:
    # For each vertex (vx, vy) and each normal (nx, ny):
    # nx*vx + ny*vy - nx*cx - ny*cy <= h
    # -nx*vx - ny*vy + nx*cx + ny*cy <= h
    
    for vx, vy in vertices:
        for nx, ny in normals:
            # n . v - n . c <= h  =>  -nx*cx - ny*cy - h <= -n . v
            A.append([-nx, -ny, -1.0])
            b.append(-(vx * nx + vy * ny))
            
            # -n . v + n . c <= h  =>  nx*cx + ny*cy - h <= n . v
            A.append([nx, ny, -1.0])
            b.append(vx * nx + vy * ny)
            
    # h >= 0
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c = [0.0, 0.0, 1.0] # Minimize h
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            cx, cy, h = res.x
            # h is the apothem (distance from center to side)
            # side_length = 2 * h / sqrt(3)
            side = 2 * h / math.sqrt(3)
            return cx, cy, side
        else:
            return None
    except Exception:
        return None

def _generate_lattice_layout(n_hexes):
    """
    Generate a compact hexagonal lattice layout for n_hexes.
    """
    sqrt3 = math.sqrt(3)
    # Lattice vectors
    u = np.array([sqrt3, 0.0])
    v = np.array([sqrt3 / 2, 1.5])
    
    candidates = []
    # Search range
    for i in range(-4, 5):
        for j in range(-4, 5):
            pos = i * u + j * v
            dist_sq = np.dot(pos, pos)
            candidates.append((dist_sq, pos))
    
    candidates.sort(key=lambda x: x[0])
    
    positions = []
    for dist_sq, pos in candidates:
        positions.append(pos)
        if len(positions) == n_hexes:
            break
            
    return positions

def _objective(params, positions_init, inner_angle):
    """
    Objective function for optimization.
    params: [x1, y1, ..., x11, y11, outer_angle, outer_cx, outer_cy]
    """
    if time.time() > _DEADLINE - 5.0:
        return 1e10
        
    n = N
    pos_params = params[:2*n]
    outer_angle = params[2*n]
    outer_cx = params[2*n + 1]
    outer_cy = params[2*n + 2]
    
    # Reconstruct positions
    positions = []
    for i in range(n):
        px = pos_params[2*i]
        py = pos_params[2*i + 1]
        positions.append((px, py))
        
    # Check non-overlap (penalty)
    penalty = 0.0
    min_dist = math.sqrt(3) - 1e-6 # Slightly less to allow touching
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < min_dist:
                penalty += (min_dist - d)**2
                
    if penalty > 1e-6:
        return 1e10 + penalty

    # Calculate enclosing hexagon side
    all_vertices = []
    for px, py in positions:
        verts = _get_hex_vertices(px, py, inner_angle)
        all_vertices.extend(verts)
        
    # We pass outer_cx, outer_cy as initial guess? 
    # The LP finds the optimal center for a given angle.
    # So we only need to optimize outer_angle.
    # The center is determined by the LP.
    res = _find_min_enclosing_hex_side(all_vertices, outer_angle)
    if res is None:
        return 1e10
        
    side = res[2]
    return side

def _polish_layout(positions, inner_angle, deadline_margin=10.0):
    """
    Polish a specific layout using Nelder-Mead.
    """
    n = N
    # Initial params: positions + outer_angle (center is found by LP)
    # We optimize positions and outer angle.
    # To reduce variables, we can fix positions initially and just optimize angle.
    # But allowing positions to move might help if the lattice isn't perfectly optimal.
    
    # Strategy 1: Optimize only outer angle (fast)
    def obj_angle(angle_params):
        angle = angle_params[0]
        all_vertices = []
        for px, py in positions:
            verts = _get_hex_vertices(px, py, inner_angle)
            all_vertices.extend(verts)
        res = _find_min_enclosing_hex_side(all_vertices, angle)
        if res is None:
            return 1e10
        return res[2]
        
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0,0)
    
    # Coarse scan
    for a in np.arange(0, 30, 0.5):
        if time.time() > _DEADLINE - deadline_margin:
            break
        s = obj_angle([a])
        if s < best_side:
            best_side = s
            best_angle = a
            # Get center from LP
            all_vertices = []
            for px, py in positions:
                verts = _get_hex_vertices(px, py, inner_angle)
                all_vertices.extend(verts)
            res = _find_min_enclosing_hex_side(all_vertices, a)
            if res:
                best_center = (res[0], res[1])
                
    # Fine scan
    if best_side < float('inf'):
        for a in np.arange(max(0, best_angle - 0.5), min(30, best_angle + 0.5) + 0.01, 0.01):
            if time.time() > _DEADLINE - deadline_margin:
                break
            s = obj_angle([a])
            if s < best_side:
                best_side = s
                best_angle = a
                all_vertices = []
                for px, py in positions:
                    verts = _get_hex_vertices(px, py, inner_angle)
                    all_vertices.extend(verts)
                res = _find_min_enclosing_hex_side(all_vertices, a)
                if res:
                    best_center = (res[0], res[1])
                    
    # Nelder-Mead polish on angle and positions
    if time.time() < _DEADLINE - deadline_margin:
        # Initial guess for positions is the lattice positions
        x0 = []
        for px, py in positions:
            x0.append(px)
            x0.append(py)
        x0.append(best_angle)
        
        # We only optimize positions and angle. Center is derived.
        # But _objective needs to return center? No, it returns side.
        # We need to update best_center after optimization.
        
        def obj_full(params):
            n = N
            pos_params = params[:2*n]
            outer_angle = params[2*n]
            
            positions_opt = []
            for i in range(n):
                px = pos_params[2*i]
                py = pos_params[2*i + 1]
                positions_opt.append((px, py))
                
            # Penalty for overlap
            penalty = 0.0
            min_dist = math.sqrt(3) - 1e-6
            for i in range(n):
                for j in range(i+1, n):
                    dx = positions_opt[i][0] - positions_opt[j][0]
                    dy = positions_opt[i][1] - positions_opt[j][1]
                    d = math.sqrt(dx*dx + dy*dy)
                    if d < min_dist:
                        penalty += (min_dist - d)**2
                        
            if penalty > 1e-6:
                return 1e10 + penalty

            all_vertices = []
            for px, py in positions_opt:
                verts = _get_hex_vertices(px, py, inner_angle)
                all_vertices.extend(verts)
                
            res = _find_min_enclosing_hex_side(all_vertices, outer_angle)
            if res is None:
                return 1e10
            return res[2]

        try:
            res_nm = minimize(obj_full, x0, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res_nm.success:
                side = res_nm.fun
                if side < best_side:
                    # Extract new positions and angle
                    new_positions = []
                    for i in range(n):
                        new_positions.append((res_nm.x[2*i], res_nm.x[2*i+1]))
                    new_angle = res_nm.x[2*n]
                    
                    # Recalculate center
                    all_vertices = []
                    for px, py in new_positions:
                        verts = _get_hex_vertices(px, py, inner_angle)
                        all_vertices.extend(verts)
                    res_lp = _find_min_enclosing_hex_side(all_vertices, new_angle)
                    if res_lp:
                        best_side = side
                        best_center = (res_lp[0], res_lp[1])
                        # Update positions to the optimized ones
                        positions = new_positions
                        best_angle = new_angle
        except Exception:
            pass
            
    return best_side, best_center, best_angle, positions

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate layouts
    layouts = []
    
    # Layout 1: Compact lattice
    pos1 = _generate_lattice_layout(N)
    layouts.append(pos1)
    
    # Layout 2: Slightly rotated lattice? Or different subset?
    # The "1-3-5-3-1" minus 2 corners might be better.
    # Let's create a specific manual layout for 11 based on symmetry.
    # Center (0,0)
    # 6 neighbors
    # 4 more at (2,0), (0,2), (-2,2), (-2,0) ? No, that's asymmetric.
    # Let's stick to the sorted lattice points, which is a good heuristic for compactness.
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    # Try different inner hexagon orientations
    # Due to symmetry of the cluster and outer hexagon, 0 and 30 are the main candidates.
    inner_angles = [0.0, 30.0]
    
    for layout_idx, positions in enumerate(layouts):
        for inner_angle in inner_angles:
            if time.time() > _DEADLINE - 10.0:
                break
                
            result = _polish_layout(positions, inner_angle, deadline_margin=10.0)
            if result:
                side, center, outer_angle, final_positions = result
                if side < best_side:
                    best_side = side
                    best_inner = [[p[0], p[1], inner_angle] for p in final_positions]
                    best_center = center
                    best_outer_angle = outer_angle
                    best_inner_angle = inner_angle
                    
    if best_inner is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    return best_inner, best_center, best_side, best_outer_angle
