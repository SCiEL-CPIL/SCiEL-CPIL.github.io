import math
import time
import numpy as np
from scipy.optimize import linprog

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

def _min_distance_hexagons(theta_diff_deg):
    """
    Calculate minimum distance between centers of two unit hexagons 
    with relative rotation theta_diff_deg to avoid overlap.
    """
    # Normalize angle to [0, 30]
    delta = theta_diff_deg % 60.0
    if delta > 30.0:
        delta = 60.0 - delta
    
    # If delta is 0, distance is sqrt(3) (inradius * 2)
    # If delta is 30, the "point" of one touches the "side" of the other?
    # Let's compute it numerically for accuracy
    # Vertices of hex1 (angle 0)
    v1 = _hex_vertices(0, 0, 0.0)
    # Vertices of hex2 (angle delta)
    v2 = _hex_vertices(0, 0, delta)
    
    min_dist = float('inf')
    # Check all pairs of vertices
    for p1 in v1:
        for p2 in v2:
            d = np.linalg.norm(p1 - p2)
            if d < min_dist:
                min_dist = d
                
    # Check vertex of 1 to edge of 2, and vertex of 2 to edge of 1
    # Edges of hex2
    edges2 = []
    for i in range(6):
        edges2.append((v2[i], v2[(i+1)%6]))
        
    for p1 in v1:
        for p_start, p_end in edges2:
            # Project p1 onto segment p_start-p_end
            v = p_end - p_start
            w = p1 - p_start
            c1 = np.dot(w, v)
            if c1 <= 0:
                d = np.linalg.norm(p1 - p_start)
            else:
                c2 = np.dot(v, v)
                if c2 <= c1:
                    d = np.linalg.norm(p1 - p_end)
                else:
                    b = c1 / c2
                    v_p = p_start + b * v
                    d = np.linalg.norm(p1 - v_p)
            if d < min_dist:
                min_dist = d
                
    # Edges of hex1
    edges1 = []
    for i in range(6):
        edges1.append((v1[i], v1[(i+1)%6]))
        
    for p2 in v2:
        for p_start, p_end in edges1:
            v = p_end - p_start
            w = p2 - p_start
            c1 = np.dot(w, v)
            if c1 <= 0:
                d = np.linalg.norm(p2 - p_start)
            else:
                c2 = np.dot(v, v)
                if c2 <= c1:
                    d = np.linalg.norm(p2 - p_end)
                else:
                    b = c1 / c2
                    v_p = p_start + b * v
                    d = np.linalg.norm(p2 - v_p)
            if d < min_dist:
                min_dist = d
                
    return min_dist

# Precompute minimum distances for a grid of angles
# Angles from 0 to 30 deg, step 0.5 deg
_ANGLE_GRID = np.arange(0, 30.25, 0.5)
_MIN_DIST_TABLE = np.array([_min_distance_hexagons(a) for a in _ANGLE_GRID])

def _get_min_dist(theta_diff_deg):
    delta = theta_diff_deg % 60.0
    if delta > 30.0:
        delta = 60.0 - delta
    # Interpolate
    idx = int(delta / 0.5)
    if idx >= len(_MIN_DIST_TABLE) - 1:
        return _MIN_DIST_TABLE[-1]
    frac = (delta - idx * 0.5) / 0.5
    return _MIN_DIST_TABLE[idx] * (1 - frac) + _MIN_DIST_TABLE[idx + 1] * frac

def _check_collision(positions, angles):
    """Check if configuration has collisions."""
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            theta_diff = angles[i] - angles[j]
            min_dist_req = _get_min_dist(theta_diff)
            if dist < min_dist_req - 1e-6:
                return False
    return True

def _min_enclosing_hex_side_lp(positions, angles, outer_angle):
    """
    Calculate exact minimum side length of a regular hexagon 
    with center (cx, cy) and rotation outer_angle that encloses all inner hexagons.
    Returns (side, cx, cy).
    """
    all_vertices = []
    for i in range(N):
        v = _hex_vertices(positions[i, 0], positions[i, 1], angles[i])
        all_vertices.append(v)
    all_vertices = np.vstack(all_vertices)
    
    base_angle = math.radians(outer_angle + 30)
    angles_n = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles_n), np.sin(angles_n)])
    
    M = len(all_vertices)
    vdots = all_vertices @ normals.T
    
    # Variables: cx, cy, R
    # Constraints: |normal_j . (v_i - center)| <= R for all i, j
    # normal_j . v_i - normal_j . center <= R
    # -normal_j . v_i + normal_j . center <= R  => normal_j . center - R <= normal_j . v_i
    # Actually: 
    # n_j . (v_i - c) <= R  => n_j . v_i - n_j . c <= R => n_j . v_i - n_j . cx - n_j . cy - R <= 0
    # -(n_j . (v_i - c)) <= R => -n_j . v_i + n_j . c <= R => -n_j . v_i + n_j . cx + n_j . cy - R <= 0
    
    A_rows = []
    b_vals = []
    
    for j in range(6):
        nx, ny = normals[j]
        # n_j . v_i - n_j . cx - n_j . cy - R <= 0
        # v_i . n_j - cx * nx - cy * ny - R <= 0
        # Since v_i is fixed, this is linear in cx, cy, R.
        # But we have M vertices.
        # We can just use the max/min projection?
        # Yes, for a given orientation, the minimal enclosing hexagon is determined by the extreme projections.
        # However, the center can move.
        # The condition is: max_i (n_j . (v_i - c)) <= R and min_i (n_j . (v_i - c)) >= -R
        # max_i (n_j . v_i) - n_j . c <= R
        # n_j . c - min_i (n_j . v_i) <= R
        
        max_proj = np.max(vdots[:, j])
        min_proj = np.min(vdots[:, j])
        
        # n_j . c - max_proj >= -R  => -n_j . c + R <= -max_proj
        # n_j . c - min_proj <= R   => n_j . c - R <= min_proj
        
        # Let's stick to the vertex-by-vertex formulation to be safe, or use the aggregated one.
        # Aggregated is much faster (12 constraints vs 12*M).
        # -nx * cx - ny * cy + R <= -max_proj
        # nx * cx + ny * cy - R <= min_proj
        
        A_rows.append([-nx, -ny, 1.0])
        b_vals.append(-max_proj)
        A_rows.append([nx, ny, -1.0])
        b_vals.append(min_proj)
        
    A = np.vstack(A_rows)
    b = np.array(b_vals)
    
    # R >= 0
    A = np.vstack([A, np.array([[0.0, 0.0, -1.0]])])
    b = np.concatenate([b, [0.0]])
    
    c_obj = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs', options={'presolve': True, 'dual_feasibility_tolerance': 1e-12, 'primal_feasibility_tolerance': 1e-12})
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return side, cx, cy
    except Exception:
        pass
    return float('inf'), 0, 0

def _build_positions_hinge(hinge_angles):
    """
    Build positions and absolute angles for 11 hexagons based on hinge angles.
    Topology: 
    Center (0)
    Ring 1 (1-6) around Center
    Ring 2 (7-10) attached to Ring 1
    
    Hinge angles: relative rotations between connected hexagons.
    We define the structure as a tree or graph.
    Let's use a specific topology:
    0: Center
    1-6: Neighbors of 0
    7: Neighbor of 1 (and 2?) - Let's say attached to 1
    8: Neighbor of 3
    9: Neighbor of 5
    10: Neighbor of 2
    
    This is getting complicated to parameterize generally.
    Instead, let's use the "Flower" layout as a base and allow local deformations?
    Or better: Use the known optimal structure for 11 hexagons.
    The optimal packing of 11 unit hexagons is likely the "hexagonal number" shape 1+6+4 (a hexagon of side 2 with 2 corners removed? No, side 2 is 19 hexagons).
    Side 2 hexagon contains 19 unit hexagons.
    11 hexagons fit in a smaller container.
    
    Let's try a different strategy:
    1. Generate a good initial layout (e.g., 1 center, 6 ring, 4 outer).
    2. Optimize the global rotation of the container.
    3. Optimize the positions and angles of the inner hexagons using L-BFGS-B with a penalty for collisions and the exact LP for the container size.
    
    This is what the current program does, but it might be slow or stuck.
    The inspiration says "LP-optimized outer hexagon" and "continuous angle search".
    
    Let's refine the optimization:
    - Fix the center of the cluster at (0,0) by subtracting the centroid? No, the container center is optimized.
    - We will optimize: positions (2N), angles (N), container angle (1).
    - Objective: Container side length + Penalty for collisions.
    - To make it faster, we can use the aggregated LP constraints inside the objective? No, LP is discrete.
    - We can use the "smooth max" approximation for the container size in the objective, then polish with exact LP.
    
    Let's implement the smooth objective again but more carefully.
    """
    pass

def _soft_max(x, beta=50.0):
    max_x = np.max(x)
    shifted = x - max_x
    return max_x + np.log(np.sum(np.exp(beta * shifted))) / beta

def _objective(params, beta=50.0, penalty_weight=1000.0):
    if time.time() > _DEADLINE - 2.0:
        return 1e6
        
    positions = params[:N*2].reshape(N, 2)
    angles = params[N*2:N*3]
    outer_angle = params[N*3]
    
    # Calculate vertices
    all_vertices = []
    for i in range(N):
        v = _hex_vertices(positions[i, 0], positions[i, 1], angles[i])
        all_vertices.append(v)
    all_vertices = np.vstack(all_vertices)
    
    # Calculate outer hexagon size (smooth approximation)
    base_angle = math.radians(outer_angle + 30)
    angles_n = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles_n), np.sin(angles_n)])
    
    vdots = all_vertices @ normals.T
    widths = np.max(vdots, axis=0) - np.min(vdots, axis=0)
    R_approx = _soft_max(widths, beta) / 2.0
    side_approx = 2.0 * R_approx * _INVSQRT3
    
    # Collision penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            theta_diff = angles[i] - angles[j]
            min_dist_req = _get_min_dist(theta_diff)
            if dist < min_dist_req:
                penalty += (min_dist_req - dist) ** 2
    
    return side_approx + penalty_weight * penalty

def _optimize_layout(initial_positions, initial_angles, initial_outer_angle):
    if time.time() > _DEADLINE - 10.0:
        return None
        
    x0 = np.concatenate([initial_positions.flatten(), initial_angles, [initial_outer_angle]])
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((0, 60))
    bounds.append((0, 60))
    
    try:
        from scipy.optimize import minimize
        res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
        
        if res.success:
            opt_positions = res.x[:N*2].reshape(N, 2)
            opt_angles = res.x[N*2:N*3]
            opt_outer_angle = res.x[N*3]
            
            if not _check_collision(opt_positions, opt_angles):
                return None
                
            side, cx, cy = _min_enclosing_hex_side_lp(opt_positions, opt_angles, opt_outer_angle)
            
            if side < float('inf'):
                opt_positions = opt_positions - np.array([cx, cy])
                cx, cy = 0.0, 0.0
                
                return {
                    'positions': opt_positions,
                    'angles': opt_angles,
                    'side': side,
                    'outer_angle': opt_outer_angle,
                    'center': (cx, cy)
                }
    except Exception:
        pass
    return None

def _generate_initial_layouts():
    layouts = []
    
    # Layout 1: Flower (1 center, 6 ring, 4 more)
    pos1 = np.array([[0,0]])
    ang1 = np.array([0.0])
    for k in range(6):
        theta = k * 60.0
        x = _SQRT3 * math.cos(math.radians(theta))
        y = _SQRT3 * math.sin(math.radians(theta))
        pos1 = np.vstack([pos1, [[x, y]]])
        ang1 = np.append(ang1, 0.0)
    # Add 4 more from ring 2
    # Positions in axial coordinates for a hex grid:
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # We need 4 more. Let's pick (2,0), (1,1), (0,2), (1,-1)?
    # Convert axial (q, r) to cartesian:
    # x = _SQRT3 * (q + r/2)
    # y = 1.5 * r
    ring2_axial = [(2,0), (1,1), (0,2), (1,-1)]
    for q, r in ring2_axial:
        x = _SQRT3 * (q + r/2.0)
        y = 1.5 * r
        pos1 = np.vstack([pos1, [[x, y]]])
        ang1 = np.append(ang1, 0.0)
    layouts.append((pos1, ang1, 0.0))
    
    # Layout 2: Random
    np.random.seed(42)
    pos2 = np.random.uniform(-5, 5, (11, 2))
    ang2 = np.random.uniform(0, 60, 11)
    layouts.append((pos2, ang2, 0.0))
    
    return layouts

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    best_result = None
    best_side = float('inf')
    
    initial_layouts = _generate_initial_layouts()
    
    for layout_idx, (pos, ang, outer_ang) in enumerate(initial_layouts):
        if time.time() > _DEADLINE - 15.0:
            break
            
        for start_outer_angle in [0.0, 15.0, 30.0]:
            if time.time() > _DEADLINE - 15.0:
                break
                
            result = _optimize_layout(pos, ang, start_outer_angle)
            if result:
                if result['side'] < best_side:
                    best_side = result['side']
                    best_result = result
                    
    if best_result is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    inner = []
    for i in range(N):
        inner.append([best_result['positions'][i][0], best_result['positions'][i][1], best_result['angles'][i]])
        
    return inner, best_result['center'], best_result['side'], best_result['outer_angle']
