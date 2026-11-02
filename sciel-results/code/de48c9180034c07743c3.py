"""
Optimized hexagon packing: 11 unit hexagons.
Strategy:
1. Generate candidate inner cluster configurations (7 central + 4 from ring 2).
2. For each configuration, find the optimal outer hexagon (center, size, angle).
3. Use a two-stage optimization:
   a. Coarse search over outer hexagon angles to find the best region.
   b. Fine-grained search using scipy.optimize.minimize on the angle, 
      where the objective function is the minimum side length S(angle) 
      computed via Linear Programming.
4. Return the best configuration found.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _lp_min_enclosing_hex(vertices, oangle_deg):
    """
    Find the minimum side length and center of a regular hexagon with fixed rotation oangle_deg
    that encloses all given vertices.
    Returns (cx, cy, side) or None if infeasible.
    """
    a_rad = math.radians(oangle_deg)
    
    # Normals for a regular hexagon with rotation a_rad.
    # The sides are perpendicular to these normals.
    # For a flat-top hexagon (0 deg), normals are at 30, 90, 150, 210, 270, 330.
    # Rotated by a_rad: normals are at 30+a, 90+a, etc.
    normals = []
    for k in range(6):
        theta = math.radians(30 + 60 * k) + a_rad
        normals.append((math.cos(theta), math.sin(theta)))
    
    n_vertices = len(vertices)
    n_constraints = n_vertices * 6
    
    # Objective: minimize S
    c_obj = np.array([0.0, 0.0, 1.0])
    
    # Constraints:
    # For each vertex v and normal n:
    # n . (v - c) <= S * SQRT3 / 2
    # -n . c - (SQRT3/2)S <= -n . v
    A_ub = np.zeros((n_constraints, 3))
    b_ub = np.zeros(n_constraints)
    
    idx = 0
    for vx, vy in vertices:
        for nx, ny in normals:
            A_ub[idx, 0] = -nx
            A_ub[idx, 1] = -ny
            A_ub[idx, 2] = -SQRT3 / 2.0
            b_ub[idx] = -(nx * vx + ny * vy)
            idx += 1
            
    bounds = [(None, None), (None, None), (0.0, None)]
    
    try:
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.status == 0:
            return result.x[0], result.x[1], result.x[2]
    except Exception:
        pass
    return None

def _get_cluster_vertices(positions, angles):
    """Get all vertices of the inner hexagons."""
    verts = []
    for (x, y), ang in zip(positions, angles):
        v = _hex_vertices(x, y, 1.0, ang)
        verts.extend(v)
    return verts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Axial coordinates for the 7-hexagon flower
    # Center (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    axial_positions = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)
    ]
    
    # Ring 2 positions
    ring2_axial = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
        (0, 2), (1, 2)
    ]
    
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return (x, y)
    
    # Base positions (7 hexagons)
    base_xy = [axial_to_xy(q, r) for q, r in axial_positions]
    base_angles = [0.0] * 7
    
    best_side = float('inf')
    best_config = None
    
    # Try all combinations of 4 from ring 2
    # There are 14 ring 2 hexagons. C(14,4) = 1001.
    # We can prune symmetric ones if needed, but 1001 is small.
    
    for combo in itertools.combinations(range(14), 4):
        if time.time() > deadline - 2.0:
            break
            
        selected_r2 = [ring2_axial[i] for i in combo]
        full_axial = axial_positions + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        full_angles = [0.0] * 11
        
        # Get vertices
        verts = _get_cluster_vertices(full_xy, full_angles)
        
        # Coarse search for outer angle
        best_angle = 0.0
        best_side_coarse = float('inf')
        
        # The optimal angle is likely near 0, 30, 60, etc.
        # Let's search in steps of 5 degrees first.
        for ang in range(0, 180, 5):
            res = _lp_min_enclosing_hex(verts, ang)
            if res:
                cx, cy, s = res
                if s < best_side_coarse:
                    best_side_coarse = s
                    best_angle = ang
        
        # Fine-grained search around best_angle
        # Define objective function for minimization
        def obj_func(angle_arr):
            ang = angle_arr[0]
            res = _lp_min_enclosing_hex(verts, ang)
            if res:
                return res[2]
            return 1e9
        
        # Use Nelder-Mead or L-BFGS-B
        # Nelder-Mead is robust for small dimensional problems
        try:
            res_min = minimize(obj_func, [best_angle], method='Nelder-Mead', 
                               options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 500})
            if res_min.success and res_min.fun < best_side_coarse:
                best_angle = res_min.x[0]
                best_side_coarse = res_min.fun
        except Exception:
            pass
            
        # Final verification and update
        final_res = _lp_min_enclosing_hex(verts, best_angle)
        if final_res:
            cx, cy, s = final_res
            if s < best_side:
                best_side = s
                best_config = {
                    'positions': full_xy,
                    'angles': full_angles,
                    'center': (cx, cy),
                    'side': s,
                    'angle': best_angle
                }
                
    if best_config is None:
        # Fallback: just the 7-hexagon flower, scaled up? No, must be 11.
        # Use the first combo found.
        # This should not happen.
        pass
        
    # Construct output
    inner = [[p[0], p[1], a] for p, a in zip(best_config['positions'], best_config['angles'])]
    center = best_config['center']
    side = best_config['side']
    angle = best_config['angle']
    
    return inner, center, side, angle
