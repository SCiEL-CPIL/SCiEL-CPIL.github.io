import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with orientation outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    # Normals to the 6 sides of the outer hexagon
    # Vertices of the outer hexagon are at angles outer_angle + 60k.
    # The sides are perpendicular to the normals.
    # The normal to the side between vertices k and k+1 is at angle outer_angle + 30 + 60k.
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    # Variables: [cx, cy, R], minimize R
    # Constraints for each vertex v and normal n:
    #   (v - c).n <= R  =>  -n.c - R <= -v.n
    #   (c - v).n <= R  =>   n.c - R <=  v.n
    
    A_rows = []
    b_vals = []
    
    for k in range(6):
        nx, ny = normals[k]
        # v_dot_n for all vertices in this normal direction
        v_dots = verts[:, 0] * nx + verts[:, 1] * ny
        
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
            # R is the inradius (distance from center to side).
            # Side length s = R / cos(30) = R / (sqrt(3)/2) = 2R/sqrt(3)
            side = 2.0 * R / _SQRT3
            return cx, cy, side
    except Exception:
        pass
    return None

def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    # Use a tiny epsilon to ensure disjointness if needed, but LP handles containment.
    # For packing, we want them to touch. Standard hex packing distance is sqrt(3) for side 1.
    scale = 1.0
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
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
        # Row-based: 3-4-4 pattern
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
        # 4-3-4 pattern
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
    elif layout_idx == 3:
        # 3-5-3 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 4:
        # 2-4-2-3 pattern
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    else:
        # Fallback: compact grid
        coords = []
        for i in range(-3, 4):
            for j in range(-3, 4):
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
    
    num_layouts = 5
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Try inner angles 0 and 30 degrees
        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 15.0:
                break
            
            all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
            
            best_local_side = float('inf')
            best_local_center = (0.0, 0.0)
            best_local_angle = 0.0
            
            # Search outer angles from 0 to 30 degrees in steps of 0.5
            # Due to symmetry, 0-30 is sufficient for regular hexagons
            for outer_angle_deg in np.arange(0, 30.1, 0.5):
                if time.time() > _DEADLINE - 15.0:
                    break
                
                res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_local_side:
                        best_local_side = side
                        best_local_center = (cx, cy)
                        best_local_angle = outer_angle_deg
            
            # Refine around the best local angle with finer steps
            if best_local_angle > 0 and best_local_angle < 30:
                refine_start = max(0, best_local_angle - 0.5)
                refine_end = min(30, best_local_angle + 0.5)
                for outer_angle_deg in np.arange(refine_start, refine_end + 0.01, 0.05):
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
    
    if best_inner is None:
        # Fallback
        positions = layouts[0]
        inner_angle = 0.0
        all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
        res = _min_enclosing_hex_side_lp(all_vertices, 0.0)
        if res:
            cx, cy, side = res
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_center = (cx, cy)
            best_side = side
            best_outer_angle = 0.0
        else:
            # Emergency fallback
            best_inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
            best_center = (0.0, 0.0)
            best_side = 40.0
            best_outer_angle = 0.0
            
    return best_inner, best_center, best_side, best_outer_angle
