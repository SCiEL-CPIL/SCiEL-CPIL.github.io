import math
import time
import numpy as np
from scipy.optimize import linprog, minimize_scalar

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

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
    Returns (center_x, center_y, side_length).
    """
    # Normals of the outer hexagon
    # If vertices are at angles theta + 60k, the edges are between theta and theta+60.
    # The normal to the edge between theta and theta+60 is theta+30.
    # So normals are at outer_angle_deg + 30 + 60*k.
    
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    # LP Variables: [cx, cy, R]
    # Minimize R
    # Constraints:
    # For each vertex v and normal n:
    # (v - c) . n <= R  =>  c . n - v . n >= -R  =>  -c . n - R <= -v . n
    # (c - v) . n <= R  =>  c . n - v . n <= R   =>   c . n - R <=  v . n
    
    A = []
    b = []
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            
            # Constraint 1: -nx*cx - ny*cy - R <= -v_dot_n
            A.append([-nx, -ny, -1.0])
            b.append(-v_dot_n)
            
            # Constraint 2: nx*cx + ny*cy - R <= v_dot_n
            A.append([nx, ny, -1.0])
            b.append(v_dot_n)
            
    # R >= 0
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3) / 2)
            return cx, cy, side
        else:
            return None
    except:
        return None

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Generate 11 positions for a compact 3-4-3 cluster
    sqrt3 = math.sqrt(3)
    scale = 1.000001 # Slightly larger than touching to avoid numerical issues in LP if needed, though LP handles equality well
    
    # Lattice vectors
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # 3-4-3 Cluster Coordinates
    # Row 1 (y=1.5): 3 hexes
    # Row 2 (y=0.0): 4 hexes
    # Row 3 (y=-1.5): 3 hexes
    
    positions = []
    # Row 1
    for i in [-1, 0, 1]:
        x = i * u[0]
        y = 1.5 * scale
        positions.append((x, y))
        
    # Row 2
    for i in [-1.5, -0.5, 0.5, 1.5]:
        x = i * u[0]
        y = 0.0
        positions.append((x, y))
        
    # Row 3
    for i in [-1, 0, 1]:
        x = i * u[0]
        y = -1.5 * scale
        positions.append((x, y))
        
    # Note: The above construction might not be perfectly centered or symmetric in x.
    # Let's center the cluster at (0,0) by subtracting the centroid.
    cx_mean = sum(p[0] for p in positions) / N
    cy_mean = sum(p[1] for p in positions) / N
    positions = [(p[0] - cx_mean, p[1] - cy_mean) for p in positions]
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # We only need to search for the relative angle between inner and outer.
    # Fix inner angle to 0. Search outer angle in [0, 30).
    inner_angle = 0.0
    
    # Get all vertices for this inner angle
    all_vertices = []
    for px, py in positions:
        verts = _get_hex_vertices(px, py, inner_angle)
        all_vertices.extend(verts)
        
    def objective(outer_angle_deg):
        if time.time() > _DEADLINE - 5.0:
            return float('inf')
        res = _find_min_enclosing_hex_side(all_vertices, outer_angle_deg)
        if res:
            return res[2]
        return float('inf')
        
    # Use minimize_scalar to find the optimal outer angle
    # The function is periodic with period 60, but symmetric in 30.
    try:
        res_opt = minimize_scalar(objective, bounds=(0, 30), method='bounded', options={'xatol': 1e-6})
        if res_opt.success:
            best_outer_angle = res_opt.x
            best_side = res_opt.fun
            # Retrieve center for this optimal angle
            res_center = _find_min_enclosing_hex_side(all_vertices, best_outer_angle)
            if res_center:
                best_center = (res_center[0], res_center[1])
            else:
                # Fallback if retrieval fails
                best_center = (0.0, 0.0)
        else:
            # Fallback to grid search if minimize_scalar fails
            for outer_angle_deg in np.arange(0, 30.1, 0.1):
                if time.time() > _DEADLINE - 5.0:
                    break
                res = _find_min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_side:
                        best_side = side
                        best_outer_angle = outer_angle_deg
                        best_center = (cx, cy)
    except Exception as e:
        # Fallback to grid search on exception
        for outer_angle_deg in np.arange(0, 30.1, 0.1):
            if time.time() > _DEADLINE - 5.0:
                break
            res = _find_min_enclosing_hex_side(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_outer_angle = outer_angle_deg
                    best_center = (cx, cy)

    if best_inner is None or best_side == float('inf'):
        # Fallback if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
    
    return best_inner, best_center, best_side, best_outer_angle
