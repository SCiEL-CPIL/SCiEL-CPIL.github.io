import time
import math
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
T0 = time.time()
DEADLINE = T0 + 145.0  # Safety margin before 150s kill
SQRT3 = math.sqrt(3.0)

def _hex_vertices_local(theta):
    """Unit hexagon vertices (circumradius 1) rotated by theta."""
    ang = np.arange(6) * (math.pi / 3.0) + theta
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)

def _outer_side_normals(theta_outer):
    """Outward normals of the 6 sides of the outer hexagon."""
    ang = theta_outer + np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)

def _get_all_vertices(centers, theta):
    """Get global vertices of all inner hexagons."""
    verts_local = _hex_vertices_local(theta)
    # centers: (N, 2), verts_local: (6, 2)
    # Result: (N*6, 2)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    return verts

def _min_enclosing_hex_lp(centers, theta, theta_outer):
    """
    Use LP to find the optimal outer hexagon center and inradius
    for given inner hexagons (fixed theta) and outer angle.
    Returns (cx, cy, side_length) or None.
    """
    all_vertices = _get_all_vertices(centers, theta)
    n_pts = len(all_vertices)
    
    # Variables: cx, cy, t (inradius)
    # Minimize t
    c = [0.0, 0.0, 1.0]
    
    A_ub = []
    b_ub = []
    
    normals = _outer_side_normals(theta_outer)
    
    for nx, ny in normals:
        # (vx - cx)*nx + (vy - cy)*ny <= t  =>  -nx*cx - ny*cy - t <= -nx*vx - ny*vy
        # -(vx - cx)*nx - (vy - cy)*ny <= t =>  nx*cx + ny*cy - t <= nx*vx + ny*vy
        
        # Vector for each point
        # We need to add constraints for all points
        # To be efficient, we can construct the matrices for all points at once if we use numpy, 
        # but linprog takes lists. Given N=11, 6 verts/hex = 66 points. 6 normals * 2 * 66 = 792 constraints.
        # This is small enough for a loop or vectorized construction.
        
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)
            
    # t >= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)

    try:
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, t = res.x
            side = t / (SQRT3 / 2.0)
            return cx, cy, side
    except Exception:
        pass
    return None

def _pairwise_penalty(centers, threshold=1.9):
    """
    Penalty for pairs of centers closer than 'threshold'.
    We use a threshold < 2.0 to allow interlocking, but penalize strongly if they get too close.
    """
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    dpair = d[iu]
    viol = np.maximum(0.0, threshold - dpair)
    return np.sum(viol * viol)

def _objective(params, theta_outer):
    """
    Objective: minimize outer side length (via LP) + penalty for overlaps.
    Note: Calling LP in objective function is expensive.
    Alternative: Use a fast approximation for the objective, and use LP only for final evaluation and maybe coarse guidance.
    However, the LP is quite fast (792 constraints). Let's try it.
    If too slow, we fall back to max projection.
    """
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    
    # Fast approximation for optimization speed:
    # Max projection onto fixed normals
    verts = _get_all_vertices(centers, theta)
    normals = _outer_side_normals(theta_outer)
    proj = verts @ normals.T
    apothem_approx = proj.max()
    s_approx = apothem_approx * 2.0 / SQRT3
    
    pen = _pairwise_penalty(centers, threshold=1.9)
    
    return s_approx + 100.0 * pen

def _hexagons_intersect(v1, v2):
    """
    Check if two convex polygons intersect using SAT (Separating Axis Theorem).
    v1, v2: (6, 2) arrays of vertices.
    """
    # Edges of v1
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    # Normals to edges
    normals1 = np.column_stack([-edges1[:, 1], edges1[:, 0]])
    normals2 = np.column_stack([-edges2[:, 1], edges2[:, 0]])
    
    all_normals = np.vstack([normals1, normals2])
    
    for n in all_normals:
        # Project both polygons onto the normal
        proj1 = v1 @ n
        proj2 = v2 @ n
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def _is_valid(centers, theta, theta_outer, cx, cy, side, tol=1e-5):
    """
    Rigorous validity check.
    1. Check all pairs of inner hexagons for intersection.
    2. Check all vertices are inside the outer hexagon.
    """
    # 1. Non-overlap
    verts_list = []
    verts_local = _hex_vertices_local(theta)
    for i in range(N):
        v = centers[i] + verts_local
        verts_list.append(v)
        
    for i in range(N):
        for j in range(i + 1, N):
            if _hexagons_intersect(verts_list[i], verts_list[j]):
                return False
                
    # 2. Containment
    all_vertices = _get_all_vertices(centers, theta)
    normals = _outer_side_normals(theta_outer)
    
    # Shift vertices relative to outer center
    rel_verts = all_vertices - np.array([cx, cy])
    
    proj = rel_verts @ normals.T
    apothem = side * SQRT3 / 2.0
    
    if proj.max() > apothem + tol:
        return False
        
    return True

def _make_hex_grid_init():
    """Hexagonal lattice: center + 6 ring + 4 outer."""
    centers = np.zeros((N, 2))
    centers[0] = [0, 0]
    for i in range(6):
        ang = i * math.pi / 3.0
        centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
    for i in range(4):
        ang = (i + 0.5) * math.pi / 3.0
        centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
    return centers

def _make_compact_init():
    """Compact cluster arrangement (triangular lattice subset)."""
    centers = np.zeros((N, 2))
    sqrt3 = math.sqrt(3.0)
    positions = [
        (0, 0),
        (2, 0), (-1, sqrt3), (1, sqrt3),
        (-1, -sqrt3), (1, -sqrt3),
        (3, 0), (-2, sqrt3), (2, sqrt3),
        (-2, -sqrt3), (2, -sqrt3)
    ]
    for i in range(N):
        centers[i] = positions[i]
    return centers

def _make_random_init(rng):
    """Random initialization with jitter around hex grid."""
    base = _make_hex_grid_init()
    jitter = rng.uniform(-0.3, 0.3, size=(N, 2))
    centers = base + jitter
    return centers

def _optimize_single(init_params, theta_o, maxiter=200):
    """Optimize using L-BFGS-B."""
    try:
        res = minimize(_objective, init_params, args=(theta_o,),
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
        return res.x
    except Exception:
        return init_params

def _polish(params, theta_o):
    """High-precision polish."""
    best_params = params
    best_obj = _objective(params, theta_o)
    
    # SLSQP polish
    try:
        res = minimize(_objective, params, args=(theta_o,),
                       method='SLSQP',
                       options={'maxiter': 300, 'ftol': 1e-14})
        if res.fun < best_obj - 1e-10:
            best_params = res.x
            best_obj = res.fun
    except Exception:
        pass
    
    return best_params

def construct():
    global T0
    
    best_score = float('inf')
    best_result = None # (inner_list, center, side, angle_deg)
    
    # We will try a few outer rotations. 
    # Since the outer hexagon can rotate and translate, the "optimal" outer rotation is coupled with the inner positions.
    # However, due to 6-fold symmetry of the outer hexagon, we only need to search 0..30 degrees for the outer rotation 
    # if we fix the inner configuration's symmetry, but here inner is free.
    # Actually, we can just fix theta_o to 0 and let the LP find the center. 
    # Wait, the outer hexagon shape is fixed (regular), but its orientation matters relative to the inner cluster.
    # If we fix theta_o = 0, we are restricting the outer hexagon to be axis-aligned.
    # Is it better to allow theta_o to vary? Yes.
    # But varying theta_o in the optimization loop is expensive because it changes the LP constraints.
    # Strategy:
    # 1. Optimize inner positions for a fixed theta_o.
    # 2. Use LP to find optimal outer center and size for that theta_o.
    # 3. Try a few theta_o values.
    
    outer_rotations = [0.0, math.pi/6, math.pi/3, math.pi/4] # 0, 30, 60, 45 deg
    init_modes = ['hex_grid', 'compact', 'random']
    
    rng = np.random.RandomState(42)
    
    for theta_o in outer_rotations:
        if time.time() > DEADLINE - 15:
            break
            
        for mode in init_modes:
            if time.time() > DEADLINE - 15:
                break
                
            if mode == 'hex_grid':
                centers = _make_hex_grid_init()
            elif mode == 'compact':
                centers = _make_compact_init()
            else:
                centers = _make_random_init(rng)
            
            theta_init = rng.uniform(-math.pi/6, math.pi/6)
            init_params = np.concatenate([centers.reshape(-1), [theta_init]])
            
            # Optimize
            params = _optimize_single(init_params, theta_o, maxiter=300)
            
            centers_p = params[:N * 2].reshape(N, 2)
            theta_p = params[N * 2]
            
            # Use LP to get precise outer hexagon
            lp_res = _min_enclosing_hex_lp(centers_p, theta_p, theta_o)
            if lp_res is None:
                continue
                
            cx, cy, s = lp_res
            
            # Check validity
            if _is_valid(centers_p, theta_p, theta_o, cx, cy, s):
                if s < best_score:
                    best_score = s
                    best_result = (centers_p.copy(), theta_p, theta_o, cx, cy, s)
            
            # Polish if promising
            if s < best_score + 0.1 and time.time() < DEADLINE - 5:
                polished = _polish(params, theta_o)
                centers_pol = polished[:N * 2].reshape(N, 2)
                theta_pol = polished[N * 2]
                
                lp_res_pol = _min_enclosing_hex_lp(centers_pol, theta_pol, theta_o)
                if lp_res_pol is not None:
                    cx_pol, cy_pol, s_pol = lp_res_pol
                    if _is_valid(centers_pol, theta_pol, theta_o, cx_pol, cy_pol, s_pol):
                        if s_pol < best_score:
                            best_score = s_pol
                            best_result = (centers_pol.copy(), theta_pol, theta_o, cx_pol, cy_pol, s_pol)

    # If no valid result found, fallback to a known valid configuration (hex grid)
    if best_result is None:
        centers = _make_hex_grid_init()
        theta = 0.0
        theta_o = 0.0
        lp_res = _min_enclosing_hex_lp(centers, theta, theta_o)
        if lp_res:
            cx, cy, s = lp_res
            inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
            return inner, (float(cx), float(cy)), float(s), float(math.degrees(theta_o))
        else:
            # Extremely unlikely
            return [[0,0,0] for _ in range(N)], (0,0), 10.0, 0.0

    centers, theta, theta_o, cx, cy, s = best_result
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (float(cx), float(cy)), float(s), float(math.degrees(theta_o))
