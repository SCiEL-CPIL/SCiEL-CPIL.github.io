import math
import time
import numpy as np
from scipy.optimize import linprog, minimize_scalar

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
    # Normals to the 6 sides of the outer hexagon
    # Vertices at outer_angle + 60k => edge normals at outer_angle + 30 + 60k
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0

    # Variables: [cx, cy, R], minimize R
    # Constraints for each vertex v and normal n:
    #   (v - c).n <= R  =>  -n.c - R <= -v.n
    #   (c - v).n <= R  =>   n.c - R <=  v.n
    
    # Vectorized construction of A and b
    # verts: (M, 2), normals: (6, 2)
    # v_dot_n: (M, 6)
    v_dot_n = verts @ normals.T
    
    # We need to form (2*M*6, 3) matrix A and (2*M*6,) vector b
    # For each pair (i, k):
    #   row1: [-nx_k, -ny_k, -1] <= - (v_i . n_k)
    #   row2: [nx_k, ny_k, -1] <=   (v_i . n_k)
    
    # To be efficient, we can construct A and b using broadcasting or simple loops if M is small.
    # M = 66, 6 normals. 66*6*2 = 792 constraints. This is small for HiGHS.
    
    A_rows = []
    b_vals = []
    
    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]
        
        # Row 1: -nx*cx - ny*cy - R <= -v_dot
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)
        
        # Row 2: nx*cx + ny*cy - R <= v_dot
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
            side = R / (_SQRT3/2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _refine_angle(verts, angle0, tol=1e-8):
    """
    Refine the outer angle around angle0 to minimize enclosing hex side length.
    Uses minimize_scalar with tight bounds, then does an ultra-fine LP sweep.
    """
    def f(angle_deg):
        res = _min_enclosing_hex_side(verts, angle_deg)
        if res is None:
            return 1e10
        return res[2]
    
    # First, bracket around angle0 with a small window
    # Symmetry: hexagon has 60° periodicity, but we work in [0, 30]
    lo = max(0.0, angle0 - 0.5)
    hi = min(30.0, angle0 + 0.5)
    
    try:
        res_scalar = minimize_scalar(f, bounds=(lo, hi), method='bounded',
                                     options={'xatol': 1e-10, 'maxiter': 200})
        if res_scalar.success:
            best_angle = res_scalar.x
        else:
            best_angle = angle0
    except Exception:
        best_angle = angle0
    
    # Ultra-fine LP sweep around best_angle
    best_side = f(best_angle)
    best_center = _min_enclosing_hex_side(verts, best_angle)
    
    # Sweep with very fine steps
    step = 0.001
    sweep_lo = max(0.0, best_angle - 0.01)
    sweep_hi = min(30.0, best_angle + 0.01)
    
    for a in np.arange(sweep_lo, sweep_hi + step/2, step):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side(verts, a)
        if res is not None:
            s = res[2]
            if s < best_side:
                best_side = s
                best_angle = a
                best_center = res
    
    # Final ultra-tight sweep
    if time.time() < _DEADLINE - 12.0:
        step2 = 0.0001
        sweep_lo2 = max(0.0, best_angle - 0.001)
        sweep_hi2 = min(30.0, best_angle + 0.001)
        for a in np.arange(sweep_lo2, sweep_hi2 + step2/2, step2):
            if time.time() > _DEADLINE - 10.0:
                break
            res = _min_enclosing_hex_side(verts, a)
            if res is not None:
                s = res[2]
                if s < best_side:
                    best_side = s
                    best_angle = a
                    best_center = res
    
    return best_angle, best_center, best_side

def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    # Use a small gap to ensure strict disjointness
    scale = 1.001 
    u = (_SQRT3*scale, 0.0)
    v = (_SQRT3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
        target = [(0,0)]
        # Ring 1
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        # Ring 2 - pick 4 that keep it compact
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        # Pick 4 closest to center
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        
        positions = []
        for ti, tj in target:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x,y))
        return positions[:N]
    elif layout_idx == 1:
        # Row-based: 3 rows of 4,1,4 or similar
        # Try: row at y=-1.5: 3 hexes, y=0: 4 hexes, y=1.5: 4 hexes
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*_SQRT3*scale + _SQRT3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k*_SQRT3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 3):
            x = k*_SQRT3*scale + _SQRT3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        # 4-3-4 but shifted
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*_SQRT3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*_SQRT3*scale + _SQRT3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 2):
            x = k*_SQRT3*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        # Fallback: sorted by distance
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    layouts = [_build_positions(i) for i in range(4)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Only inner angle 0 is usually optimal for hexagons on a lattice, 
        # but we check 30 just in case.
        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 10.0:
                break
            
            all_vertices = []
            for px, py in positions:
                all_vertices.extend(_hex_vertices(px, py, inner_angle))
            all_vertices = np.array(all_vertices)
            
            # Coarse search
            best_coarse_side = float('inf')
            best_coarse_angle = 0.0
            best_coarse_center = (0.0, 0.0)
            
            # Symmetry allows searching 0 to 30 degrees
            # Step 1.0 degree
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
            if best_coarse_angle > 0 or best_coarse_side < float('inf'):
                # Refine around best coarse angle
                # Ensure we stay within [0, 30]
                start_angle = max(0.0, best_coarse_angle - 1.0)
                end_angle = min(30.0, best_coarse_angle + 1.0)
                
                for outer_angle_deg in np.arange(start_angle, end_angle + 0.05, 0.05):
                    if time.time() > _DEADLINE - 10.0:
                        break
                    res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < best_coarse_side:
                            best_coarse_side = side
                            best_coarse_angle = outer_angle_deg
                            best_coarse_center = (cx, cy)
            
            # Use the dedicated refinement function for high-precision polish
            if best_coarse_side < float('inf') and time.time() < _DEADLINE - 12.0:
                refined_angle, refined_center, refined_side = _refine_angle(
                    all_vertices, best_coarse_angle)
                if refined_side < best_coarse_side:
                    best_coarse_side = refined_side
                    best_coarse_angle = refined_angle
                    best_coarse_center = refined_center
            
            if best_coarse_side < best_side:
                best_side = best_coarse_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_coarse_center
                best_outer_angle = best_coarse_angle
                best_inner_angle = inner_angle
    
    # Final global polish: re-run refinement on the best configuration found
    if best_inner is not None and time.time() < _DEADLINE - 15.0:
        # Reconstruct vertices from best_inner
        all_vertices = []
        for row in best_inner:
            px, py, ang = row[0], row[1], row[2]
            all_vertices.extend(_hex_vertices(px, py, ang))
        all_vertices = np.array(all_vertices)
        
        refined_angle, refined_center, refined_side = _refine_angle(
            all_vertices, best_outer_angle)
        if refined_side < best_side:
            best_side = refined_side
            best_outer_angle = refined_angle
            best_center = refined_center
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
