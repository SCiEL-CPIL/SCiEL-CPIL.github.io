import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin before 120s hard limit

def _get_unit_hex_verts():
    """Vertices of a unit regular hexagon centered at origin, angle 0."""
    # Side length 1.
    # Vertices at angles 0, 60, 120, ...
    # Distance from center to vertex (circumradius) = 1 for side 1?
    # No, for a regular hexagon, side length = circumradius.
    # So vertices are at distance 1 from center.
    angles = np.arange(6) * np.pi / 3
    return np.column_stack((np.cos(angles), np.sin(angles)))

UNIT_VERTS = _get_unit_hex_verts()

def _transform_hex_verts(positions, angles):
    """
    positions: (N, 2) array
    angles: (N,) array in radians
    Returns: (N, 6, 2) array of vertices
    """
    N = len(positions)
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        theta = angles[i]
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + positions[i]
    return verts

def _check_disjoint_fast(verts, tol=1e-9):
    """
    Check if all hexagons are disjoint using SAT (Separating Axis Theorem).
    verts: (N, 6, 2)
    Returns True if disjoint.
    """
    N = len(verts)
    # Get edges for each hexagon
    # Edge k is from vert k to vert (k+1)%6
    # Normal to edge k is perpendicular to edge vector, pointing outward.
    # For a convex polygon, if there is a separating axis, it is normal to one of the edges.
    
    # Precompute edge normals for each hexagon
    # Edge vector: e = v_{k+1} - v_k
    # Normal: n = (e_y, -e_x) normalized? No, we just need the direction.
    # Project all vertices of both hexagons onto the normal and check for gap.
    
    for i in range(N):
        for j in range(i + 1, N):
            # Check axes from hex i and hex j
            # Axes from hex i
            for k in range(6):
                v1 = verts[i, k]
                v2 = verts[i, (k + 1) % 6]
                edge = v2 - v1
                # Normal vector (perpendicular)
                n = np.array([edge[1], -edge[0]])
                # Project all vertices of hex i and hex j onto n
                proj_i = np.dot(verts[i], n)
                proj_j = np.dot(verts[j], n)
                
                min_i = np.min(proj_i)
                max_i = np.max(proj_i)
                min_j = np.min(proj_j)
                max_j = np.max(proj_j)
                
                # If intervals [min_i, max_i] and [min_j, max_j] do not overlap, they are disjoint
                if max_i < min_j - tol or max_j < min_i - tol:
                    break # Found a separating axis for this pair, move to next pair
            else:
                # No separating axis found from hex i, check hex j
                for k in range(6):
                    v1 = verts[j, k]
                    v2 = verts[j, (k + 1) % 6]
                    edge = v2 - v1
                    n = np.array([edge[1], -edge[0]])
                    proj_i = np.dot(verts[i], n)
                    proj_j = np.dot(verts[j], n)
                    
                    min_i = np.min(proj_i)
                    max_i = np.max(proj_i)
                    min_j = np.min(proj_j)
                    max_j = np.max(proj_j)
                    
                    if max_i < min_j - tol or max_j < min_i - tol:
                        break # Found separating axis
                else:
                    # No separating axis found, they overlap
                    return False
    return True

def _min_enclosing_hex_side(points, outer_angle_rad):
    """
    Find min side length of regular hexagon with given orientation containing all points.
    points: (M, 2) array
    outer_angle_rad: float
    Returns: (center_x, center_y, side_length) or None
    """
    # Normals of the outer hexagon
    # The sides of a regular hexagon are perpendicular to vectors at angles:
    # phi + 30, phi + 90, ... ?
    # Let's define the hexagon by its circumradius R.
    # Vertices are at R * (cos(phi + k*pi/3), sin(phi + k*pi/3)).
    # The supporting lines (sides) are perpendicular to vectors at angles phi + pi/6 + k*pi/3.
    # Let's use the normals to the sides.
    # If the hexagon is "pointy topped" (vertex at top), the normals are at 90, 30, -30, -90, -150, 150.
    # Let's assume the angle `outer_angle_rad` defines the rotation of the hexagon itself.
    # The normals to the 6 sides are at angles: outer_angle_rad + pi/6 + k*pi/3 for k=0..5.
    
    normals = []
    for k in range(6):
        a = outer_angle_rad + math.pi / 6 + k * math.pi / 3
        normals.append((math.cos(a), math.sin(a)))
    normals = np.array(normals) # (6, 2)
    
    # We want to find c (2D) and R (scalar) such that:
    # For all p in points, for all n in normals:
    # (p - c) . n <= R
    # (c - p) . n <= R
    # This is equivalent to:
    # c . n - p . n <= R
    # -c . n + p . n <= R
    
    # Variables: c_x, c_y, R
    # Objective: minimize R => c = [0, 0, 1]
    
    A = []
    b = []
    
    # points is (M, 2)
    # normals is (6, 2)
    # We need to construct constraints for each point and each normal.
    
    # Vectorized construction
    # p . n for all p and n: (M, 6)
    p_dot_n = points @ normals.T
    
    # Constraint 1: c . n - p . n <= R  =>  c . n - R <= p . n
    # c . n is [nx * cx + ny * cy]
    # So: nx * cx + ny * cy - R <= p_dot_n
    
    # Constraint 2: -c . n + p . n <= R  =>  -nx * cx - ny * cy - R <= -p_dot_n
    
    for k in range(6):
        nx, ny = normals[k]
        # For each point i:
        # nx * cx + ny * cy - R <= p_dot_n[i, k]
        A.append([nx, ny, -1.0])
        b.append(np.max(p_dot_n[:, k])) # Wait, the constraint must hold for ALL points.
        # Actually, linprog requires A_ub @ x <= b_ub.
        # If we just take the max, we ensure it holds for the "worst" point.
        # Is that sufficient?
        # (c - p) . n <= R for all p.
        # Max_p ((c - p) . n) <= R.
        # (c . n - Min_p(p . n)) <= R.
        # So: c . n - R <= Min_p(p . n).
        
        # Let's re-derive.
        # We need (c - p) . n <= R for all p.
        # This is equivalent to c . n - p . n <= R.
        # The tightest constraint is for the p that maximizes (c . n - p . n), which is p that minimizes p . n.
        # So: c . n - R <= min(p . n).
        
        # Similarly for the other side: (p - c) . n <= R => p . n - c . n <= R.
        # Tightest for p maximizing p . n.
        # So: -c . n - R <= -max(p . n).
        
        min_p_dot_n = np.min(p_dot_n[:, k])
        max_p_dot_n = np.max(p_dot_n[:, k])
        
        # c . n - R <= min_p_dot_n
        A.append([nx, ny, -1.0])
        b.append(min_p_dot_n)
        
        # -c . n - R <= -max_p_dot_n
        A.append([-nx, -ny, -1.0])
        b.append(-max_p_dot_n)
        
    # R >= 0
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    A = np.array(A)
    b = np.array(b)
    c_obj = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, 
                     bounds=[(None, None), (None, None), (0, None)],
                     method='highs')
        if res.success:
            cx, cy, R = res.x
            # Side length s is related to R (circumradius) by s = R for regular hexagon?
            # Yes, for a regular hexagon, side length = circumradius.
            side = R
            return cx, cy, side
        else:
            return None
    except:
        return None

def _objective(params):
    """
    Objective function for optimization.
    params: (3*N,) array [x1, y1, th1, ..., xN, yN, thN]
    Returns: side length + penalty
    """
    if time.time() > _DEADLINE - 5.0:
        return 1e10
        
    N = len(params) // 3
    positions = params[0:3*N:3].reshape(N, 2)
    angles = params[1:3*N:3]
    
    verts = _transform_hex_verts(positions, angles)
    
    # Check disjointness
    if not _check_disjoint_fast(verts):
        # Penalty: sum of overlaps? Or just a large number.
        # For smoothness, we could estimate overlap, but for now, large penalty.
        return 1e6 + np.sum(np.abs(params)) * 0.01
    
    # Collect all vertices for enclosing hexagon
    all_verts = verts.reshape(-1, 2)
    
    # Find best outer angle
    best_side = float('inf')
    # Search outer angle from 0 to 30 degrees (symmetry)
    # Coarse step
    for outer_angle_deg in np.arange(0, 30.1, 1.0):
        res = _min_enclosing_hex_side(all_verts, math.radians(outer_angle_deg))
        if res:
            side = res[2]
            if side < best_side:
                best_side = side
    
    return best_side

def construct():
    global _START_TIME, _DEADLINE
    
    # Fallback
    fallback_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    fallback_return = (fallback_inner, (0.0, 0.0), 40.0, 0.0)
    
    if time.time() > _DEADLINE - 10.0:
        return fallback_return

    # Initial guess: Hexagonal lattice
    sqrt3 = math.sqrt(3)
    # Unit hexagons on a triangular lattice.
    # Distance between centers of touching hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    # Wait, side length 1 hexagon has apothem sqrt(3)/2.
    # Distance between centers of adjacent hexagons in a tiling is 2 * apothem = sqrt(3).
    
    # Lattice vectors:
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    # Note: 1.5 = 3/2.
    
    u = np.array([sqrt3, 0.0])
    v = np.array([sqrt3 / 2, 1.5])
    
    # Generate a grid of lattice points
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * u + j * v
            lattice_points.append(p)
            
    # Sort by distance to origin
    lattice_points.sort(key=lambda p: np.dot(p, p))
    init_pos = np.array(lattice_points[:N])
    init_ang = np.zeros(N)
    
    init_params = np.zeros(3 * N)
    for i in range(N):
        init_params[3*i] = init_pos[i, 0]
        init_params[3*i+1] = init_pos[i, 1]
        init_params[3*i+2] = init_ang[i]
        
    # Optimize
    best_side = float('inf')
    best_params = init_params
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # 1. Differential Evolution for global search
    # Bounds: positions within [-10, 10], angles [0, 2pi]
    bounds = [(-10, 10), (-10, 10), (0, 2*math.pi)] * N
    
    # Check time
    if time.time() < _DEADLINE - 30.0:
        try:
            from scipy.optimize import differential_evolution
            # Use a small population and few generations to stay within time
            res = differential_evolution(_objective, bounds, 
                                         maxiter=50, 
                                         popsize=10, 
                                         seed=42,
                                         tol=1e-4,
                                         polish=False) # Disable polish to control time
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception as e:
            pass
    
    # 2. Nelder-Mead for local polish
    if time.time() < _DEADLINE - 10.0:
        try:
            from scipy.optimize import minimize
            # Start from best found
            x0 = best_params
            res = minimize(_objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
    # Final verification and calculation of outer hexagon
    if best_side > 1e5:
        return fallback_return

    N = len(best_params) // 3
    positions = best_params[0:3*N:3].reshape(N, 2)
    angles = best_params[1:3*N:3]
    
    verts = _transform_hex_verts(positions, angles)
    all_verts = verts.reshape(-1, 2)
    
    # Find the specific outer hexagon parameters for the best inner configuration
    final_best_side = float('inf')
    final_center = (0.0, 0.0)
    final_outer_angle = 0.0
    
    # Fine search for outer angle
    for outer_angle_deg in np.arange(0, 30.1, 0.1):
        res = _min_enclosing_hex_side(all_verts, math.radians(outer_angle_deg))
        if res:
            side = res[2]
            if side < final_best_side:
                final_best_side = side
                final_center = (res[0], res[1])
                final_outer_angle = outer_angle_deg
                
    # If the optimization failed to find a valid packing, use the initial lattice
    if final_best_side > 1e5:
         # Try to validate the initial lattice
         init_verts = _transform_hex_verts(init_pos, init_ang)
         if _check_disjoint_fast(init_verts):
             init_all_verts = init_verts.reshape(-1, 2)
             for outer_angle_deg in np.arange(0, 30.1, 0.5):
                 res = _min_enclosing_hex_side(init_all_verts, math.radians(outer_angle_deg))
                 if res:
                     side = res[2]
                     if side < final_best_side:
                         final_best_side = side
                         final_center = (res[0], res[1])
                         final_outer_angle = outer_angle_angle
                         positions = init_pos
                         angles = init_ang
                         break
                         
    if final_best_side > 1e5:
        return fallback_return

    # Format output
    inner = []
    for i in range(N):
        inner.append([positions[i, 0], positions[i, 1], math.degrees(angles[i])])
        
    return inner, final_center, final_best_side, final_outer_angle

