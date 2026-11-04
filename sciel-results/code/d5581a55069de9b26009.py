import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0 # Safety margin for 150s limit

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    """
    Returns penetration depth if overlapping (negative), or separation distance if not.
    Uses Separating Axis Theorem for convex polygons.
    """
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1) % len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap  # positive = separated
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap  # negative = overlapping

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    # Normals to the sides are at 30 + 60k degrees relative to the vertex directions.
    # If vertices are at rot_deg, sides normals are at rot_deg + 30 + 60k.
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    # Apothem = (sqrt(3)/2) * side
    # max_proj is the max distance from center to a side (apothem)
    # So side = 2 * apothem / sqrt(3)
    return (2.0 / np.sqrt(3.0)) * max_proj

def smooth_overlap_penalty(params, beta=50.0):
    """
    Compute smooth overlap penalty.
    """
    xys_rots = params[:3*N].reshape(N, 3)
    centers = xys_rots[:, :2]
    
    # Precompute vertices for all hexagons
    vertices_list = []
    for i in range(N):
        cx, cy, rot = xys_rots[i]
        v = get_hex_vertices(cx, cy, 1.0, rot)
        vertices_list.append(v)
    
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.0:  # Only compute SAT if close enough
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penetrations.append(-d_sat)
    
    if len(penetrations) == 0:
        return 0.0
    
    p_arr = np.array(penetrations)
    # Smooth max approximation of sum of squares
    # Using soft-max on squared values
    p2 = p_arr**2
    max_p2 = np.max(p2)
    if max_p2 < 1e-10:
        return 0.0
    scaled = 2.0 * p_arr / beta
    max_s = np.max(scaled)
    if max_s > 500:
        return np.sum(p_arr**2)
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return beta * lse

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        # Use smooth penalty
        penalty = smooth_overlap_penalty(params, beta=50.0)
        
        # Compute outer side length
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        L = outer_hex_side(all_points, outer_rot)
        
        return L + 100.0 * penalty

    def make_hex_cluster():
        """
        Creates a compact cluster of 12 hexagons:
        1 center, 6 in first ring, 5 in second ring (missing one to make 12).
        """
        pts = []
        sqrt3 = np.sqrt(3)
        # Center
        pts.append([0.0, 0.0, 0.0])
        # Ring 1 (distance sqrt(3))
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang), 0.0])
        # Ring 2 (distance 2*sqrt(3)) - we need 5 more
        # The vertices of the hexagon formed by ring 1 centers are at distance 2*sqrt(3) from center?
        # No, the "corners" of the hexagonal grid are at distance 2*sqrt(3) * cos(30) ? 
        # Let's use standard hex grid coordinates.
        # Axial coords (q, r).
        # Center (0,0).
        # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2) ... wait
        # Let's just place them manually based on geometry.
        # The 6 neighbors of the center form a ring.
        # The "outer" hexagons sit in the "valleys" between ring 1 hexagons.
        # There are 6 such valleys. We pick 5.
        # Distance from center to valley center: 2 * sqrt(3) * cos(30 deg)? No.
        # The distance between centers of adjacent hexagons is sqrt(3).
        # The distance from center to the center of a hexagon in the "corner" of the 1+6 cluster is 2*sqrt(3).
        # Let's verify.
        # Center (0,0). Neighbor at (sqrt(3), 0).
        # Neighbor at (sqrt(3)/2, 1.5).
        # The "corner" hexagon would be at (sqrt(3)/2 + sqrt(3)*cos(30), 1.5 + sqrt(3)*sin(30))?
        # Actually, the centers of the ring 2 hexagons that touch two ring 1 hexagons are at distance 2*sqrt(3) from the origin.
        # Angles: 30, 90, 150, 210, 270, 330.
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang), 0.0])
            
        pts = np.array(pts)
        # Center the cluster
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def make_grid(rows, cols):
        """
        Creates a rectangular grid of hexagons.
        """
        pts = []
        sqrt3 = np.sqrt(3)
        dx = sqrt3
        dy = 1.5
        for r in range(rows):
            for c in range(cols):
                x = c * dx
                if r % 2 == 1:
                    x += dx / 2.0
                y = r * dy
                pts.append([x, y, 0.0])
        
        pts = np.array(pts)
        if len(pts) != N:
            # If grid size is wrong, trim or pad (shouldn't happen with correct input)
            pts = pts[:N]
            while len(pts) < N:
                pts = np.vstack([pts, [[0,0,0]]])
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def is_valid(params, tol=1e-5):
        """Check if configuration has no overlaps."""
        xys_rots = params[:3*N].reshape(N, 3)
        vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
        
        for i in range(N):
            for j in range(i+1, N):
                d = sat_overlap(vertices_list[i], vertices_list[j])
                if d < -tol:
                    return False
        return True

    def get_L(params):
        """Compute outer side length for a valid configuration."""
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        return outer_hex_side(all_points, outer_rot)

    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Compact Cluster (1+6+5)
    configs.append(make_hex_cluster())
    
    # 2. Grids
    configs.append(make_grid(2, 6))
    configs.append(make_grid(3, 4))
    configs.append(make_grid(4, 3))
    configs.append(make_grid(6, 2))
    
    # 3. Rotated variants of grids
    for base in [make_grid(2, 6), make_grid(3, 4)]:
        rot = base.copy()
        rot[:, 2] = 30.0 # Rotate all inner hexagons
        configs.append(rot)
        
    # 4. Random perturbations of the best cluster
    base_cluster = make_hex_cluster()
    for _ in range(3):
        perturbed = base_cluster.copy()
        perturbed[:, :2] += np.random.normal(0, 0.1, (N, 2))
        configs.append(perturbed)

    outer_rots = [0.0, 30.0, 15.0, 45.0, 60.0]
    
    # Optimization Loop
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Check initial validity
            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            # Run Optimization
            try:
                # L-BFGS-B is good for this
                res = minimize(
                    objective,
                    init_params,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 300,
                        'ftol': 1e-8,
                        'gtol': 1e-6,
                        'maxfun': 1000
                    }
                )
                
                # Check result
                if is_valid(res.x):
                    L_val = get_L(res.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res.x.copy()
                        
            except Exception:
                pass

            # Try Nelder-Mead for local polishing if we have a good candidate
            if best_params is not None and time.time() - start_time < DEADLINE - 15:
                try:
                    res_nm = minimize(
                        objective,
                        best_params,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 500,
                            'xatol': 1e-5,
                            'fatol': 1e-5,
                            'adaptive': True
                        }
                    )
                    if is_valid(res_nm.x):
                        L_val = get_L(res_nm.x)
                        if L_val < best_L:
                            best_L = L_val
                            best_params = res_nm.x.copy()
                except Exception:
                    pass

    if best_params is None:
        # Fallback: Linear array (very large, but valid)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = []
    for x, y, rot in xys_rots:
        inner.append([x, y, rot])
    
    return inner, (0.0, 0.0), best_L, outer_rot
