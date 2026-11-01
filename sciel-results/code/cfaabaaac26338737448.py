import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    # Normals in outer frame (30 + i*60 degrees)
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        # Transform to outer local frame
        local = (verts - np.array([ox, oy])) @ R.T
        # Project onto all 6 normals
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def get_polygon_axes(poly):
    """Get normal axes for each edge of a polygon."""
    axes = []
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        e = p2 - p1
        # Normal is perpendicular to edge
        # If edge is (dx, dy), normal is (-dy, dx) or (dy, -dx)
        # We normalize it
        norm = np.linalg.norm(e)
        if norm > 1e-9:
            n_vec = np.array([-e[1], e[0]]) / norm
            axes.append(n_vec)
    return axes

def sat_overlap_depth(p1, p2):
    """
    Calculate the minimum overlap depth between two convex polygons.
    Returns 0 if disjoint, otherwise the positive value representing
    the minimum translation distance to separate them.
    p1, p2: numpy arrays of shape (n, 2)
    """
    axes = get_polygon_axes(p1) + get_polygon_axes(p2)
    
    min_depth = float('inf')
    
    for ax in axes:
        # Project p1
        proj1 = p1 @ ax
        min1 = np.min(proj1)
        max1 = np.max(proj1)
        
        # Project p2
        proj2 = p2 @ ax
        min2 = np.min(proj2)
        max2 = np.max(proj2)
        
        # Check for overlap
        # If max1 < min2 or max2 < min1, they are disjoint along this axis
        if max1 < min2 or max2 < min1:
            return 0.0 # Disjoint
        
        # Calculate overlap depth along this axis
        # The overlap interval is [max(min1, min2), min(max1, max2)]
        # The depth is the length of this intersection
        depth = min(max1, max2) - max(min1, min2)
        
        # The actual separation distance is the minimum such depth
        # Note: This is the distance to move one poly to be just touching.
        if depth < min_depth:
            min_depth = depth
            
    return min_depth

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            depth = sat_overlap_depth(polys[i], polys[j])
            if depth > eps:
                return False
    return True

def overlap_penalty(inner, weight=1.0):
    """
    Calculate a penalty based on overlap depth.
    """
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    
    total_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            depth = sat_overlap_depth(polys[i], polys[j])
            if depth > 0:
                total_penalty += depth * weight
    return total_penalty

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Keep 5s safety margin

    # Initial configurations
    # 1. Standard Lattice (all angles 0)
    axial = [(0, 0),
             (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, -1), (1, 1), (-2, 1), (-1, -1)]
    
    # We scale the lattice slightly to ensure initial non-overlap if needed, 
    # but usually the optimizer handles it. 
    # However, starting with exact lattice positions (touching) can be problematic 
    # for gradient-based methods if they try to move them further apart.
    # Let's start with a slightly expanded lattice to give room to move.
    scale = 1.0 + 0.01 
    
    init_configs = []
    
    # Config 1: Lattice, angles 0
    init_centers = [(1.5 * scale * q, SQRT3 * scale * (r + q / 2.0)) for q, r in axial]
    init_inner_1 = [[x, y, 0.0] for x, y in init_centers]
    init_configs.append(init_inner_1)
    
    # Config 2: "Star" pattern - Rotate outer hexagons
    # Center (0,0) angle 0. 
    # Ring 1 (6 hexes): Rotate by 30 deg? 
    # Let's try rotating the 6 neighbors by 30 degrees.
    # The 4 extra hexes in the seed's topology: 
    # (2,-1), (1,1), (-2,1), (-1,-1)
    # These are adjacent to the ring.
    # Let's assign angles based on distance from center.
    init_inner_2 = []
    for (x, y) in init_centers:
        dist = math.hypot(x, y)
        if dist < 0.1:
            ang = 0.0
        else:
            ang = 30.0
        init_inner_2.append([x, y, ang])
    init_configs.append(init_inner_2)
    
    # Config 3: Random angles
    np.random.seed(42)
    init_inner_3 = []
    for (x, y) in init_centers:
        ang = np.random.uniform(-30, 30)
        init_inner_3.append([x, y, float(ang)])
    init_configs.append(init_inner_3)

    # Outer hexagon initial guess
    # Center of mass
    cx0 = sum(c[0] for c in init_centers) / len(init_centers)
    cy0 = sum(c[1] for c in init_centers) / len(init_centers)
    init_outer_center = (cx0, cy0)
    init_outer_angle = 0.0

    best_side = float('inf')
    best_inner = init_inner_1
    best_outer_center = init_outer_center
    best_outer_angle = init_outer_angle
    
    # Objective function for optimization
    # Variables: [x0, y0, a0, ..., xN-1, yN-1, aN-1, cx, cy, theta]
    def objective(params):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        
        # Calculate required side
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        
        # Calculate overlap penalty
        # We use a high weight to effectively constrain the problem
        pen = overlap_penalty(il, weight=10.0)
        
        return side + pen

    # Run local optimization for each initial config
    for k, init_inner in enumerate(init_configs):
        if time.time() > deadline - 10:
            break
            
        # Build initial params
        p0 = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(init_inner):
            p0[i * 3] = x
            p0[i * 3 + 1] = y
            p0[i * 3 + 2] = a
        p0[N * 3] = init_outer_center[0]
        p0[N * 3 + 1] = init_outer_center[1]
        p0[N * 3 + 2] = init_outer_angle

        try:
            # Nelder-Mead is good for small, non-smooth problems
            res = minimize(objective, p0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 5000})
            
            # Check result
            if res.fun < best_side:
                # Verify disjointness
                res_params = res.x
                res_inner = []
                for i in range(N):
                    res_inner.append([float(res_params[i * 3]), 
                                      float(res_params[i * 3 + 1]), 
                                      float(res_params[i * 3 + 2])])
                res_ocx = float(res_params[N * 3])
                res_ocy = float(res_params[N * 3 + 1])
                res_oa = float(res_params[N * 3 + 2])
                
                if check_disjoint(res_inner, eps=1e-4):
                    real_side = compute_required_side_fast(res_inner, (res_ocx, res_ocy), res_oa)
                    if real_side < best_side:
                        best_side = real_side
                        best_inner = res_inner
                        best_outer_center = (res_ocx, res_ocy)
                        best_outer_angle = res_oa
                        # Update t0 reference? No, keep global deadline.
                        
        except Exception as e:
            pass

    # Final Polish: High precision optimization on the best found solution
    if time.time() < deadline - 5:
        p_best = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(best_inner):
            p_best[i * 3] = x
            p_best[i * 3 + 1] = y
            p_best[i * 3 + 2] = a
        p_best[N * 3] = best_outer_center[0]
        p_best[N * 3 + 1] = best_outer_center[1]
        p_best[N * 3 + 2] = best_outer_angle

        try:
            # Very tight tolerance
            res_polish = minimize(objective, p_best, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 10000})
            
            if res_polish.fun < best_side:
                res_params = res_polish.x
                res_inner = []
                for i in range(N):
                    res_inner.append([float(res_params[i * 3]), 
                                      float(res_params[i * 3 + 1]), 
                                      float(res_params[i * 3 + 2])])
                res_ocx = float(res_params[N * 3])
                res_ocy = float(res_params[N * 3 + 1])
                res_oa = float(res_params[N * 3 + 2])
                
                if check_disjoint(res_inner, eps=1e-5):
                    real_side = compute_required_side_fast(res_inner, (res_ocx, res_ocy), res_oa)
                    if real_side < best_side:
                        best_side = real_side
                        best_inner = res_inner
                        best_outer_center = (res_ocx, res_ocy)
                        best_outer_angle = res_oa

        except Exception:
            pass

    # Ensure we return a valid result even if optimization failed
    if not check_disjoint(best_inner, eps=1e-4):
        # Fallback to initial lattice
        best_inner = init_inner_1
        best_outer_center = init_outer_center
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
