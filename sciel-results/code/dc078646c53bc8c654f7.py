"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0  # 120s safety margin within 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _compute_all_vertices(inner_hexes):
    """Precompute all vertices of inner hexagons as a numpy array (N*6, 2)."""
    n_inner = len(inner_hexes)
    verts = np.zeros((n_inner * 6, 2))
    for idx, (ix, iy, iangle) in enumerate(inner_hexes):
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + math.pi * k / 3
            verts[idx * 6 + k, 0] = ix + math.cos(a)
            verts[idx * 6 + k, 1] = iy + math.sin(a)
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg, all_vertices=None):
    """For a fixed outer hexagon angle, find optimal center and side length using LP.
    Optimized vectorized constraint construction.
    """
    if all_vertices is None:
        all_vertices = _compute_all_vertices(inner_hexes)
    
    n_pts = all_vertices.shape[0]
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg
    
    # Precompute normals for the 6 directions
    angles = np.array([math.radians(angle_deg + 30 + 60 * k) for k in range(6)])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
    
    # Project all vertices onto each normal direction
    # projs[i, k] = all_vertices[i] @ normals[k]
    projs = all_vertices @ normals.T  # (n_pts, 6)
    
    # For the LP, we need: for each direction k and vertex i:
    #   -nx_k * cx - ny_k * cy - r <= -projs[i, k]
    # Rearranged: nx_k * cx + ny_k * cy + r >= projs[i, k]
    # In standard form (<=): -nx_k * cx - ny_k * cy - r <= -projs[i, k]
    
    # Build A_ub efficiently using Kronecker-like structure
    # Row (k, i): [-nx_k, -ny_k, -1]
    # b_ub[k, i] = -projs[i, k]
    
    # b_ub: shape (6, n_pts) transposed to (n_pts, 6), then flattened
    b_ub = -projs.T.flatten()  # (6 * n_pts,)
    
    # A_ub: for each of 6 directions, repeat the row [-nx_k, -ny_k, -1] n_pts times
    neg_normals = -normals  # (6, 2)
    A_ub_parts = np.empty((6 * n_pts, 3))
    for k in range(6):
        start = k * n_pts
        end = start + n_pts
        A_ub_parts[start:end, 0] = neg_normals[k, 0]
        A_ub_parts[start:end, 1] = neg_normals[k, 1]
        A_ub_parts[start:end, 2] = -1.0
    
    # Add r >= 0: -r <= 0
    A_ub = np.vstack([A_ub_parts, [[0.0, 0.0, -1.0]]])
    b_ub = np.append(b_ub, 0.0)
    
    c = np.array([0.0, 0.0, 1.0])  # minimize r
    
    from scipy.optimize import linprog
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if result.success:
        cx_opt = result.x[0]
        cy_opt = result.x[1]
        r_opt = result.x[2]
    else:
        # Fallback: use centroid and compute required r
        cx_mean = all_vertices[:, 0].mean()
        cy_mean = all_vertices[:, 1].mean()
        centered = all_vertices - np.array([cx_mean, cy_mean])
        projs_c = centered @ normals.T  # (n_pts, 6)
        r_opt = projs_c.max()
        if r_opt < 0:
            r_opt = 0.0
        cx_opt = cx_mean
        cy_opt = cy_mean
    
    side = r_opt / (math.sqrt(3) / 2)
    return cx_opt, cy_opt, side, angle_deg

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg).
    Uses multi-resolution angle sweep with LP at each angle.
    """
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Precompute all vertices once
    all_vertices = _compute_all_vertices(inner_hexes)
    
    # Due to 6-fold symmetry of the hexagon, we search angle in [0, 60)
    
    # Pass 1: coarse sweep (30 points)
    angles_coarse = np.linspace(0, 60, 30, endpoint=False)
    for angle_deg in angles_coarse:
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Pass 2: medium sweep around best (40 points in 8-degree window)
    if best_side < float('inf'):
        lo = max(0.0, best_angle - 4.0)
        hi = min(60.0, best_angle + 4.0)
        angles_med = np.linspace(lo, hi, 40)
        for angle_deg in angles_med:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Pass 3: fine sweep (50 points in 1-degree window)
    if best_side < float('inf'):
        lo = max(0.0, best_angle - 0.5)
        hi = min(60.0, best_angle + 0.5)
        angles_fine = np.linspace(lo, hi, 50)
        for angle_deg in angles_fine:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Pass 4: ultra-fine sweep (30 points in 0.1-degree window)
    if best_side < float('inf'):
        lo = max(0.0, best_angle - 0.05)
        hi = min(60.0, best_angle + 0.05)
        angles_ultra = np.linspace(lo, hi, 30)
        for angle_deg in angles_ultra:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Pass 5: Nelder-Mead polish on angle only (LP gives optimal cx,cy)
    if best_side < float('inf') and time.time() < _DEADLINE - 15.0:
        from scipy.optimize import minimize_scalar
        
        def objective_angle(ang):
            ang = float(ang % 60.0)
            _, _, side_p, _ = _min_enclosing_hex_side_lp(inner_hexes, ang, all_vertices)
            return side_p
        
        # Local minimization around best_angle
        lo = max(0.0, best_angle - 0.1)
        hi = min(60.0, best_angle + 0.1)
        try:
            res = minimize_scalar(objective_angle, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-10, 'maxiter': 100})
            if res.success:
                ang_opt = float(res.x % 60.0)
                cx_v, cy_v, side_v, _ = _min_enclosing_hex_side_lp(inner_hexes, ang_opt, all_vertices)
                if side_v < best_side:
                    best_side = side_v
                    best_center = (cx_v, cy_v)
                    best_angle = ang_opt
        except Exception:
            pass
    
    return best_center[0], best_center[1], best_side, best_angle

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate 11 points in a hexagonal cluster arrangement
    # Central hexagon + ring of 6 + 4 more in second ring
    # Use triangular lattice with spacing sqrt(3) for edge-to-edge contact
    sqrt3 = math.sqrt(3)
    # Scale factor to ensure strict disjointness (small gap between hexagons)
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate candidate points in a hexagonal lattice around the origin
    # For 11 unit hexagons, the optimal arrangement is a central hexagon
    # surrounded by a ring of 6, plus 4 more in the second ring.
    # Using triangular lattice with spacing sqrt(3) (edge-to-edge contact).
    
    # The 11 points in a compact hexagonal cluster:
    # Ring 0: (0,0) - 1 point
    # Ring 1: 6 points at distance sqrt(3)
    # Ring 2: 4 points (not all 12, just the 4 that fit compactly)
    
    # Generate all lattice points with |i|,|j| <= 2
    candidates = []
    seen = set()
    for i in range(-2, 3):
        for j in range(-2, 3):
            # Skip duplicates due to lattice symmetry
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    # Sort by distance from origin
    candidates.sort(key=lambda t: t[0])
    
    # Take the first N points
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
    
    # Create inner hexagons with all angles = 30 (Pointy-Top)
    # This aligns the lattice vectors (0 and 60 deg) with the flat-to-flat normals,
    # requiring a minimum separation of sqrt(3) instead of 2.
    # All hexagons have the same orientation for a rigid cluster.
    inner = [[pos[0], pos[1], 30.0] for pos in positions]
    
    # Compute minimum enclosing hexagon
    if time.time() > _DEADLINE - 10.0:
        # Quick estimate
        xs = [h[0] for h in inner]
        ys = [h[1] for h in inner]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        max_dist = 0
        for h in inner:
            d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
            if d > max_dist:
                max_dist = d
        # Add circumradius 1 to max_dist for a rough bound
        side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
        return inner, (cx, cy), side, 0.0
    
    cx, cy, side, angle = _min_enclosing_hex_side(inner)
    
    # Final safety check: ensure the side length is reasonable
    # The theoretical minimum for 11 hexagons in a hexagonal cluster
    # with side-1 hexagons touching edge-to-edge:
    # The cluster has diameter ~ 3*sqrt(3) + 2 (from center to farthest vertex)
    # The enclosing hexagon side should be around 3.5-4.0
    if side < 2.0 or side > 10.0:
        # Fallback to a safe estimate
        xs = [h[0] for h in inner]
        ys = [h[1] for h in inner]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        max_dist = 0
        for h in inner:
            d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
            if d > max_dist:
                max_dist = d
        side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
        angle = 0.0
    
    return inner, (cx, cy), side, angle
# EVOLVE-BLOCK-END
