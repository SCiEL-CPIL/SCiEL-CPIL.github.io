import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _get_normals(angle_deg):
    """Normals to the sides of a hexagon with vertex at angle_deg."""
    # Vertices at angle + 60k.
    # Edges are between vertices.
    # Normal to edge between v_k and v_{k+1} is at angle + 30 + 60k.
    base_angle = math.radians(angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts.
    Returns side length or None.
    """
    normals = _get_normals(outer_angle_deg)
    M = len(verts)
    
    # Variables: [cx, cy, R]
    # Constraints: n_k . (v_i - c) <= R  =>  n_k . v_i - n_k . c <= R
    #              n_k . (c - v_i) <= R  =>  n_k . c - n_k . v_i <= R
    
    A_rows = []
    b_vals = []
    
    for k in range(6):
        n = normals[k]
        v_dots = verts @ n
        
        # n . v - n . c <= R  =>  -n[0]*cx - n[1]*cy - R <= -v_dots
        A_rows.append(np.column_stack([np.full(M, -n[0]), np.full(M, -n[1]), np.full(M, -1.0)]))
        b_vals.append(-v_dots)
        
        # n . c - n . v <= R  =>  n[0]*cx + n[1]*cy - R <= v_dots
        A_rows.append(np.column_stack([np.full(M, n[0]), np.full(M, n[1]), np.full(M, -1.0)]))
        b_vals.append(v_dots)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    # R >= 0
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)
    
    c_obj = [0.0, 0.0, 1.0]
    try:
        from scipy.optimize import linprog
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3/2)
            return side, cx, cy
    except Exception:
        pass
    return None, 0, 0

def _overlap_penalty(verts1, verts2):
    """
    Calculate penalty for overlap between two hexagons (fixed orientation).
    Uses SAT with 6 axes.
    """
    # Axes are normals of the hexagon
    normals = _get_normals(0.0)
    min_overlap = float('inf')
    
    for n in normals:
        # Project verts1
        p1 = verts1 @ n
        min1, max1 = np.min(p1), np.max(p1)
        # Project verts2
        p2 = verts2 @ n
        min2, max2 = np.min(p2), np.max(p2)
        
        # Check separation
        if max1 < min2 or max2 < min1:
            return 0.0 # Separated
        
        # Overlap depth on this axis
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    # min_overlap is the penetration depth if > 0, or separation distance if < 0?
    # If they overlap, all axes have positive overlap. The minimum positive overlap is the penetration.
    # If they are separated, at least one axis has negative overlap (separation).
    # Actually, if separated, the "overlap" calculated as min(max) - max(min) will be negative.
    # So min_overlap will be negative.
    # We want penalty for penetration (min_overlap > 0).
    if min_overlap > 0:
        return min_overlap ** 2
    else:
        return 0.0

def _objective(positions, outer_side):
    """
    Objective function for optimization.
    positions: array of shape (11, 2)
    outer_side: side length of the outer hexagon (fixed for this call)
    """
    R_outer = outer_side * _SQRT3 / 2
    normals_outer = _get_normals(0.0) # Outer hexagon orientation 0
    
    total_penalty = 0.0
    
    # Boundary constraints
    for i in range(N):
        cx, cy = positions[i]
        verts = _hex_vertices(cx, cy, 0.0)
        # Check if all vertices are inside outer hexagon
        for k in range(6):
            n = normals_outer[k]
            proj = verts @ n
            max_proj = np.max(proj)
            if max_proj > R_outer:
                total_penalty += (max_proj - R_outer) ** 2
                
    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            v1 = _hex_vertices(positions[i][0], positions[i][1], 0.0)
            v2 = _hex_vertices(positions[j][0], positions[j][1], 0.0)
            total_penalty += _overlap_penalty(v1, v2)
            
    return total_penalty

def _initial_guess():
    """Generate initial lattice positions."""
    scale = 1.0001
    u = (_SQRT3*scale, 0.0)
    v = (_SQRT3/2*scale, 1.5*scale)
    
    # 11 hexagons: center + 6 ring + 4 from second ring
    target = [(0,0)]
    ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    target.extend(ring1)
    ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
    # Pick 4 closest to center to keep it compact
    # Distances squared: (2,0)->4, (1,1)->2, (0,2)->4, (-1,2)->5, (-2,1)->5, (-2,0)->4, (-1,-1)->2, (0,-2)->4, (1,-2)->5, (2,-1)->5
    # Closest: (1,1), (-1,-1) [dist 2]
    # Next: (2,0), (0,2), (-2,0), (0,-2) [dist 4]
    # Let's pick (1,1), (-1,-1), (2,0), (0,2)
    selected_ring2 = [(1,1), (-1,-1), (2,0), (0,2)]
    target.extend(selected_ring2)
    
    positions = []
    for ti, tj in target:
        x = ti*u[0] + tj*v[0]
        y = ti*u[1] + tj*v[1]
        positions.append([x, y])
        
    return np.array(positions)

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    # Binary search on outer side length
    low = 3.0
    high = 6.0
    best_side = high
    best_positions = _initial_guess()
    
    # Try to find the minimum S such that optimization succeeds
    # We will run a few iterations of binary search
    # Each iteration runs an optimization
    
    # To save time, we can just run optimization for a fixed S that is likely good,
    # or binary search quickly.
    
    # Let's binary search
    for _ in range(10):
        if time.time() > _DEADLINE - 10.0:
            break
            
        mid = (low + high) / 2.0
        
        # Try to optimize positions for this S
        x0 = _initial_guess()
        # Add some noise to escape local minima
        x0 = x0 + np.random.normal(0, 0.01, x0.shape)
        
        try:
            res = minimize(_objective, x0, args=(mid,), method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-6})
            
            if res.success and res.fun < 1e-4:
                # Feasible
                high = mid
                best_side = mid
                best_positions = res.x
            else:
                # Not feasible
                low = mid
        except Exception:
            low = mid
            
    # Refine the best found side
    # We can try to optimize the side length directly now?
    # Or just use the best_side from binary search.
    
    # Let's do one final polish: minimize S directly?
    # No, S is discrete in the binary search.
    # We can try to reduce S slightly and re-optimize.
    
    final_side = best_side
    final_positions = best_positions
    
    # Verify and get center
    # The outer hexagon is centered at (0,0) in our optimization.
    # But we can shift the whole cluster to center it better?
    # The LP in _min_enclosing_hex_side_lp finds the optimal center.
    # Let's use it to get the exact side and center for the final positions.
    
    all_vertices = []
    for px, py in final_positions:
        all_vertices.extend(_hex_vertices(px, py, 0.0))
    all_vertices = np.array(all_vertices)
    
    # Find min enclosing hexagon for these vertices
    res = _min_enclosing_hex_side_lp(all_vertices, 0.0)
    if res[0] is not None:
        side, cx, cy = res
        # Use this side and center
        # Note: The positions are relative to (0,0) in the optimization.
        # The LP found a center (cx, cy) that encloses them.
        # We can return the positions as is, and the center as (cx, cy).
        # The evaluator will check if hexagons at `inner` are inside `outer`.
        # `inner` coordinates are absolute.
        # So we return `final_positions` as `inner`.
        # And `center` as `(cx, cy)`.
        # And `side` as `side`.
        
        inner = [[p[0], p[1], 0.0] for p in final_positions]
        return inner, (cx, cy), side, 0.0
        
    # Fallback if LP fails
    inner = [[p[0], p[1], 0.0] for p in final_positions]
    return inner, (0.0, 0.0), final_side, 0.0

