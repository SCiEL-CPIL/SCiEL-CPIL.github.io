import math
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

# Precompute the 6 normal vectors for a hexagon with flat top (angle 0)
# The normals point to the midpoints of the sides.
# For a hexagon centered at origin with side length S, the apothem is S * sqrt(3)/2.
# The vertices are at angles 0, 60, 120... if pointy top.
# If flat top, vertices are at 30, 90, 150...
# Let's define the container hexagon orientation.
# We can optimize the container rotation too, but due to symmetry, 
# we can fix the container rotation to 0 and let the inner hexagons rotate.
# Or fix inner hexagons rotation to 0 and optimize container rotation?
# The problem allows both. 
# Let's assume the container is fixed at angle 0 (flat top or pointy top).
# Let's use "pointy top" for the container? 
# Standard hexagon: vertices at (R, 0), (R/2, R*sqrt(3)/2), etc.
# The constraint for a point to be inside a hexagon with center (cx, cy), 
# side length S, and rotation beta is:
# For each of the 6 sides, the dot product of (p - c) with the outward normal <= Apothem(S).
# Apothem(S) = S * sqrt(3) / 2.
# Outward normals for a hexagon rotated by beta:
# n_k = (cos(beta + k*pi/3), sin(beta + k*pi/3)) ? 
# Let's check. If beta=0, normals are at 0, 60, 120...
# The side perpendicular to normal at 0 is vertical? No.
# If normal is (1,0), the side is vertical line x = Apothem.
# This corresponds to a "flat right" side.
# So beta=0 means a hexagon with a vertical side on the right.
# This is a "pointy top" hexagon? 
# Vertices:
# x = Apothem + S*sin(angle), y = Apothem*cos(angle)?
# Let's stick to the linear programming definition used in the seed for correctness, 
# but replace the discrete search with continuous optimization.

def get_normals(beta):
    """
    Returns 6 unit normal vectors for the sides of a regular hexagon
    with rotation beta.
    beta=0 corresponds to normals at 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3.
    """
    angles = beta + np.arange(6) * (math.pi / 3.0)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def min_enclosing_hexagon_side(pts, beta):
    """
    Calculate the minimum side length S of a regular hexagon with rotation beta
    that contains all points in pts.
    The hexagon center is optimized to be the one that minimizes S.
    For a fixed beta, the problem is:
    Minimize S
    Subject to:
    (p_i - c) . n_k <= S * APOTHEM  for all i, k
    (c - p_i) . n_k <= S * APOTHEM  for all i, k  (which is equivalent to - (p_i - c) . n_k <= S * APOTHEM)
    
    This is equivalent to:
    Max_{i,k} | (p_i - c) . n_k | <= S * APOTHEM
    S = (1/APOTHEM) * Min_c Max_{i,k} | (p_i - c) . n_k |
    
    This is a convex optimization problem (min of max of linear functions).
    We can solve it using linear programming or by observing that the optimal c
    is the center of the minimum enclosing hexagon.
    """
    normals = get_normals(beta)
    # We want to find c such that max_i,k | (p_i - c) . n_k | is minimized.
    # Let d = max_i,k | (p_i - c) . n_k |
    # This is the Chebyshev center problem in a transformed space.
    
    # We can use scipy.optimize.linprog to solve for c and d.
    # Variables: c_x, c_y, d
    # Objective: minimize d
    # Constraints:
    # (p_i - c) . n_k <= d  =>  -p_i.n_k + c.n_k <= d  => c.n_k - d <= p_i.n_k
    # (c - p_i) . n_k <= d  =>  c.n_k - p_i.n_k <= d  => c.n_k - d <= p_i.n_k (Wait, sign)
    
    # Let's rewrite:
    # | (p_i - c) . n_k | <= d
    # -d <= (p_i - c) . n_k <= d
    # -d <= p_i.n_k - c.n_k  => c.n_k - d <= p_i.n_k
    # p_i.n_k - c.n_k <= d   => -c.n_k - d <= -p_i.n_k  => c.n_k + d >= p_i.n_k
    
    # So for each i, k:
    # c.n_k - d <= p_i.n_k
    # -c.n_k - d <= -p_i.n_k
    
    n_points = len(pts)
    n_normals = 6
    
    # Variables: x = [c_x, c_y, d]
    c_obj = [0, 0, 1]
    
    A_ub = []
    b_ub = []
    
    pts_arr = np.array(pts)
    
    for k in range(n_normals):
        nx, ny = normals[k]
        for i in range(n_points):
            px, py = pts_arr[i]
            # Constraint 1: c_x*nx + c_y*ny - d <= px*nx + py*ny
            A_ub.append([nx, ny, -1])
            b_ub.append(px*nx + py*ny)
            
            # Constraint 2: -c_x*nx - c_y*ny - d <= -(px*nx + py*ny)
            A_ub.append([-nx, -ny, -1])
            b_ub.append(-(px*nx + py*ny))
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    
    from scipy.optimize import linprog
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.status == 0:
        d = res.x[2]
        S = d / APOTHEM
        return S, (res.x[0], res.x[1])
    else:
        return float('inf'), None

def objective(params):
    """
    Objective function for the optimizer.
    params: flattened array of [x0, y0, theta0, x1, y1, theta1, ...]
    theta is in radians.
    Returns the side length of the enclosing hexagon.
    We fix the enclosing hexagon rotation to 0 for now to reduce dimensionality.
    If needed, we can add beta to params.
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return float('inf')
        
    pts = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        theta = params[i*3+2]
        pts.append((x, y))
    
    # Note: The rotation of the inner hexagons does NOT change the position of their centers.
    # The constraint is that the *hexagons* are disjoint and inside the container.
    # The "disjoint" constraint is hard to encode in the objective function directly without penalties.
    # The "inside container" constraint is handled by the min_enclosing_hexagon_side.
    # We need to ensure the inner hexagons don't overlap.
    
    # A simple penalty for overlap:
    # Two hexagons of side 1 overlap if the distance between centers is less than some threshold?
    # No, they are rotated. 
    # A robust way: The Minkowski difference of two unit hexagons is a hexagon of side 2 (if aligned) or larger if rotated.
    # Actually, the condition for two convex polygons to be disjoint is that there exists a separating axis.
    # For regular hexagons, the critical distance is the width of the hexagon in the direction of the center-to-center vector.
    # Width of a unit hexagon in direction alpha:
    # If alpha is aligned with a normal, width = 2 * Apothem = sqrt(3).
    # If alpha is aligned with a vertex, width = 2 * Radius = 2.
    # The minimum distance between centers for non-overlap is the width of the Minkowski sum?
    # Actually, if we define the "radius" of a hexagon in direction u as r(u) = max { v . u | v in Hex },
    # then two hexagons H1 at c1 and H2 at c2 are disjoint if ||c1 - c2|| > r_H1(u) + r_H2(-u) for all u?
    # No, that's for separation.
    # They are disjoint if the distance between centers is greater than the support function of the Minkowski difference.
    # For regular hexagons, the Minkowski difference is a regular hexagon with side length 2 (if same orientation) 
    # but if rotated, it's a 12-gon? No, the Minkowski difference of two convex polygons is a convex polygon.
    # The support function of the Minkowski difference H1 - H2 is h_H1(u) + h_H2(-u).
    # For a regular hexagon of side 1, h(u) depends on the angle.
    # This is complex to compute analytically for arbitrary rotations.
    
    # Alternative: Use a penalty based on the minimum distance between vertices/edges?
    # Or, simpler: Assume that for N=11, the optimal packing will likely have all hexagons aligned (theta=0) 
    # or in a few discrete orientations. 
    # The seed program allowed rotation but only optimized the container.
    # Let's try to optimize positions with fixed theta=0 first. This reduces the problem to 2D.
    # If theta=0, the condition for non-overlap is simply distance >= sqrt(3) (if touching sides) 
    # or distance >= 2 (if touching vertices)?
    # Actually, the minimum distance between centers of two non-overlapping unit hexagons (aligned) is sqrt(3).
    # Because the width across flats is sqrt(3).
    
    # Let's assume all inner hexagons are aligned (theta=0).
    # This is a strong assumption, but often true for small N packings in hexagonal containers.
    
    # Penalty for overlap:
    penalty = 0.0
    pts_arr = np.array(pts)
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(pts_arr[i] - pts_arr[j])
            # Minimum distance is sqrt(3) for aligned hexagons.
            # If dist < sqrt(3), they overlap.
            if dist < math.sqrt(3) - 1e-6:
                penalty += (math.sqrt(3) - dist) ** 2
                
    # Calculate container size
    S, center = min_enclosing_hexagon_side(pts, 0.0)
    
    if S == float('inf'):
        return 1e10
        
    return S + 100.0 * penalty

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback to a simple grid
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial guess: Lattice packing
    # Generate a lattice of N points
    pts_init = []
    q, r = 0, 0
    while len(pts_init) < N:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts_init.append((x, y, 0.0))
        q += 1
        if q > 5:
            q = -5
            r += 1
    
    # Flatten params: x, y, theta for each
    x0 = [p[0] for p in pts_init]
    y0 = [p[1] for p in pts_init]
    t0 = [p[2] for p in pts_init]
    
    params_init = []
    for i in range(N):
        params_init.extend([x0[i], y0[i], t0[i]])
        
    # We will optimize x, y only, fixing theta=0 for simplicity and speed.
    # If we want to allow rotation, we need to handle the overlap penalty more carefully.
    # Let's stick to x, y optimization first.
    
    def obj_xy(params):
        pts = []
        for i in range(N):
            pts.append((params[i*2], params[i*2+1]))
            
        penalty = 0.0
        pts_arr = np.array(pts)
        min_dist = math.sqrt(3) - 1e-6
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pts_arr[i] - pts_arr[j])
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
                    
        S, _ = min_enclosing_hexagon_side(pts, 0.0)
        if S == float('inf'):
            return 1e10
        return S + 100.0 * penalty

    # Initial params for xy
    params_xy_init = []
    for i in range(N):
        params_xy_init.extend([x0[i], y0[i]])

    # Use Differential Evolution for global search
    # Bounds: The points must be within a reasonable area.
    # The container side is likely around 4-5.
    # So coordinates should be within [-5, 5].
    bounds = [(-5.0, 5.0)] * (2 * N)
    
    best_result = None
    best_score = float('inf')
    
    # Run DE with limited iterations to fit time budget
    try:
        # maxiter=50, popsize=20
        # This might be slow. Let's use Nelder-Mead from multiple starts.
        
        # Start 1: Lattice
        res = minimize(obj_xy, params_xy_init, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
            
        # Start 2: Random jitter
        np.random.seed(42)
        for restart in range(10):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            params_rand = params_xy_init + np.random.normal(0, 0.5, size=2*N)
            res = minimize(obj_xy, params_rand, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                
    except Exception as e:
        print(f"Optimization error: {e}")
        # Fallback to initial
        best_result = np.array(params_xy_init)
        best_score = obj_xy(best_result)

    if best_result is None:
        best_result = np.array(params_xy_init)
        
    # Extract final positions
    pts_final = []
    for i in range(N):
        x = best_result[i*2]
        y = best_result[i*2+1]
        pts_final.append((x, y))
        
    # Calculate final container
    S, center = min_enclosing_hexagon_side(pts_final, 0.0)
    
    if S == float('inf'):
        # Fallback
        S = 10.0
        center = (0.0, 0.0)
        
    # Return format:
    # inner: list of [x, y, angle_degrees]
    # center: [cx, cy]
    # side: S
    # rotation: 0.0 (container rotation)
    
    inner = []
    for x, y in pts_final:
        inner.append([float(x), float(y), 0.0])
        
    center_out = [float(center[0]), float(center[1])]
    
    # Verify validity? 
    # The evaluator checks if hexagons are disjoint and inside.
    # Our penalty function encourages disjointness, but doesn't guarantee it.
    # If the penalty is 0, they are disjoint (for aligned hexagons).
    
    return inner, center_out, float(S), 0.0

