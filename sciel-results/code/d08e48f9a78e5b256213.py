"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
    a = np.radians(angle_deg)
    # vertices at angles 0, 60, ..., 300 degrees from center
    angles = np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0]) * np.pi / 180.0 + a
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack([x, y])

def _hex_distance_to_origin(cx, cy, side, angle_deg):
    """Return the minimum distance from origin to the hexagon boundary (0 if origin inside).
    For a regular hexagon, the distance from center to each side (apothem) is side * sqrt(3)/2.
    The distance from center to each vertex is side.
    The closest point on the hexagon to the origin depends on direction.
    We compute the minimum distance from origin to any point on the hexagon.
    """
    # For a regular hexagon centered at (cx, cy), the distance from center to boundary
    # in direction theta is: r(theta) = apothem / cos((theta - nearest_vertex_angle) mod 60 deg)
    # Actually, for a regular hexagon, the support function gives the distance.
    # Simpler: compute distance from origin to each edge, take max of 0 and min distance to edges? No.
    # The minimum distance from origin to the hexagon = max(0, min over all edges of distance to that edge)
    # But that's not right either. Let me think again.
    # The distance from a point to a convex polygon is 0 if inside, else min distance to any edge.
    # So: if origin is inside hexagon, distance = 0. Else, min distance to any edge.
    
    # First check if origin is inside the hexagon
    verts = _hex_vertices(cx, cy, side, angle_deg)
    
    # Point-in-polygon test using winding or cross products
    # For a convex polygon, check if point is on same side of all edges
    inside = True
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        # Cross product to check side
        ex = v2[0] - v1[0]
        ey = v2[1] - v1[1]
        px = 0.0 - v1[0]
        py = 0.0 - v1[1]
        cross = ex * py - ey * px
        if i == 0:
            sign = np.sign(cross)
            if cross == 0:
                sign = 1.0
        else:
            if cross * sign < -1e-10:
                inside = False
                break
    if inside:
        return 0.0
    
    # Compute min distance from origin to any edge
    min_dist = float('inf')
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        # Project origin onto line segment v1-v2
        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]
        len2 = dx * dx + dy * dy
        if len2 < 1e-15:
            d = np.sqrt(v1[0]**2 + v1[1]**2)
        else:
            t = -(v1[0] * dx + v1[1] * dy) / len2
            t = np.clip(t, 0.0, 1.0)
            px = v1[0] + t * dx
            py = v1[1] + t * dy
            d = np.sqrt(px * px + py * py)
        if d < min_dist:
            min_dist = d
    return min_dist

def _hex_overlap_penalty(h1, h2, side1=1.0, side2=1.0):
    """Compute a penalty based on how much two hexagons overlap.
    h1, h2 are (cx, cy, angle_deg) tuples.
    Returns 0 if no overlap, positive value if overlapping.
    """
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    
    # Approximate overlap by checking if distance between centers is less than sum of inradii
    # inradius = side * sqrt(3)/2
    inr1 = side1 * np.sqrt(3) / 2
    inr2 = side2 * np.sqrt(3) / 2
    
    dx = cx2 - cx1
    dy = cy2 - cy1
    dist = np.sqrt(dx * dx + dy * dy)
    
    # If centers are far apart, no overlap
    if dist > inr1 + inr2 + side1 + side2:
        return 0.0
    
    # For a rough overlap measure, use the distance between the closest points
    # This is an approximation; for better accuracy we'd need polygon intersection
    # Let's use a simpler metric: if dist < inr1 + inr2, there's likely overlap
    # The "depth" of overlap is approximately (inr1 + inr2 - dist)
    overlap = max(0.0, (inr1 + inr2) - dist)
    return overlap ** 2

def _compute_outer_hex_side(inner_positions, side_len=1.0):
    """Given inner hexagon positions, find the minimum side length of an enclosing hexagon.
    inner_positions: list of (cx, cy, angle_deg)
    Returns the minimum side length of a regular hexagon (with any center and rotation) that encloses all.
    """
    # For each inner hexagon, compute its 6 vertices
    all_verts = []
    for cx, cy, ang in inner_positions:
        verts = _hex_vertices(cx, cy, side_len, ang)
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)  # shape (6*N, 2)
    
    # We need to find a regular hexagon (center, side, rotation) that contains all points
    # and minimizes side.
    # The enclosing hexagon's side length is determined by the maximum "support" in 6 directions.
    # For a hexagon with center (ox, oy) and rotation phi, the support in direction theta is:
    # The hexagon boundary in direction theta from center is at distance r(theta) where
    # r(theta) = apothem / cos(delta) where delta is angle from nearest vertex direction.
    # Actually, for a regular hexagon with inradius r_in = side * sqrt(3)/2,
    # the distance from center to boundary in direction theta is:
    # d(theta) = r_in / cos((theta - k*60deg) mod 60deg) for the appropriate k
    # But this is the distance to the vertex direction. Let me use a different approach.
    
    # Alternative: The hexagon can be described as the intersection of 6 half-planes.
    # For a regular hexagon with center (ox, oy), side s, rotation phi:
    # The 6 outward normals are at angles phi + 0, phi+60, ..., phi+300 degrees.
    # Each half-plane: n_i . (p - center) <= apothem, where apothem = s * sqrt(3)/2
    # Wait, that's for the inscribed circle. The actual constraint is:
    # For each face, the distance from center to the face is the apothem.
    # The face normal directions are at angles phi + 30, phi+90, ... (offset by 30 from vertices)
    # Let me be more careful.
    
    # A regular hexagon with vertices at angles phi, phi+60, ..., phi+300 from center,
    # with circumradius R = s (side length = circumradius for regular hexagon).
    # The inradius (apothem) = R * cos(30 deg) = R * sqrt(3)/2.
    # The 6 faces have outward normals at angles phi+30, phi+90, phi+150, phi+210, phi+270, phi+330.
    # Each face is at distance apothem from center along its normal.
    # So the constraint for point p to be inside: for each i, n_i . (p - center) <= apothem
    # where n_i = (cos(phi + 30 + i*60), sin(phi + 30 + i*60))
    
    # For a given (ox, oy, phi), the required apothem is:
    # apothem >= max over all points p of max over i of n_i . (p - (ox, oy))
    # And side = apothem / cos(30 deg) = apothem * 2 / sqrt(3)
    
    # So we need to minimize over (ox, oy, phi): max_p max_i n_i(phi) . (p - (ox, oy))
    
    # This is a 3-variable optimization. Let's use a simple approach:
    # For a fixed phi, the problem reduces to finding (ox, oy) that minimizes
    # max_p max_i n_i . (p - (ox, oy))
    # = max_i (max_p n_i . p - n_i . (ox, oy))
    # Let M_i = max_p n_i . p (the max projection of all points in direction n_i)
    # Then we need: M_i - n_i . (ox, oy) <= A for all i, where A is the apothem.
    # Minimize A subject to n_i . (ox, oy) >= M_i - A for all i.
    # This is a linear program in variables (ox, oy, A)!
    
    # For a fixed phi, solve the LP. Then optimize over phi.
    
    best_side = float('inf')
    
    # Try several rotations
    for phi_deg in np.arange(0, 60, 5.0):
        phi = np.radians(phi_deg)
        # 6 face normals
        normals = []
        for i in range(6):
            ang = phi + np.radians(30 + i * 60)
            normals.append((np.cos(ang), np.sin(ang)))
        normals = np.array(normals)  # (6, 2)
        
        # Compute M_i = max over all points of n_i . p
        projections = all_verts @ normals.T  # (6*N, 6)
        M = np.max(projections, axis=0)  # (6,)
        
        # LP: minimize A
        # subject to: n_i . (ox, oy) + A >= M_i for i=0..5
        # i.e., -n_i . (ox, oy) - A <= -M_i
        # Variables: x = [ox, oy, A], 3 variables
        # Constraints: for each i, [-n_ix, -n_iy, -1] . x <= -M_i
        # Also A >= 0
        
        c = np.array([0.0, 0.0, 1.0])  # minimize A
        A_ub = -normals[:, :2]  # shape (6, 2) -> need (6, 3)
        A_ub = np.hstack([-normals, -np.ones((6, 1))])
        b_ub = -M
        # A >= 0: -A <= 0
        A_ub_full = np.vstack([A_ub, [[0, 0, -1.0]]])
        b_ub_full = np.concatenate([b_ub, [0.0]])
        
        from scipy.optimize import linprog
        res = linprog(c, A_ub=A_ub_full, b_ub=b_ub_full, bounds=[(None, None), (None, None), (0, None)], method='highs')
        
        if res.success:
            A_val = res.x[2]
            side_val = A_val * 2.0 / np.sqrt(3)
            if side_val < best_side:
                best_side = side_val
    
    return best_side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    # Initial guess: arrange hexagons in a compact cluster
    # Hexagonal lattice arrangement: 1 center, 6 around, 4 more
    # For 11 hexagons, a good initial arrangement:
    # Center at (0,0)
    # 6 neighbors at distance 2*inradius = sqrt(3)
    # 4 more further out
    
    inradius = np.sqrt(3) / 2
    # Hex grid positions (in units of side length)
    # Using axial coordinates
    # For a hex grid, neighbors of (0,0) are at:
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in axial coords
    # Convert to Cartesian: x = s * (q + r/2), y = s * r * sqrt(3)/2
    # where s = side length = 1
    
    def axial_to_cartesian(q, r, s=1.0):
        x = s * (q + r / 2.0)
        y = s * r * np.sqrt(3) / 2.0
        return x, y
    
    # 11 hexagons in a compact hexagonal shape
    # Ring 0: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: pick 4 from the 12 in ring 2
    # Ring 2 positions: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)
    # Let's take: (2,0), (0,2), (-2,0), (0,-2) for symmetry
    
    hex_positions_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    
    # Convert to Cartesian
    initial_positions = []
    for q, r in hex_positions_axial:
        x, y = axial_to_cartesian(q, r)
        initial_positions.append([x, y, 0.0])
    
    initial_positions = np.array(initial_positions)
    
    # Compute initial side length
    def objective(params):
        # params: 3*N array [x1, y1, a1, x2, y2, a2, ...]
        positions = params.reshape(N, 3)
        
        # Compute outer hex side
        side = _compute_outer_hex_side(positions.tolist())
        
        # Add overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                h1 = (positions[i, 0], positions[i, 1], positions[i, 2])
                h2 = (positions[j, 0], positions[j, 1], positions[j, 2])
                penalty += _hex_overlap_penalty(h1, h2)
        
        return side + 100.0 * penalty
    
    # Initial evaluation
    init_val = objective(initial_positions.flatten())
    
    best_params = initial_positions.flatten().copy()
    best_val = init_val
    
    # Use Nelder-Mead optimization
    try:
        result = minimize(
            objective,
            initial_positions.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
        )
        
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
        
        # Check deadline
        if time.time() > deadline - 20.0:
            pass  # Skip further optimization
        else:
            # Try a few random restarts
            for restart in range(3):
                if time.time() > deadline - 10.0:
                    break
                # Perturb initial solution
                perturbed = initial_positions + np.random.randn(N, 3) * 0.5
                perturbed[:, 2] *= 0.1  # Small angle perturbations
                
                try:
                    res2 = minimize(
                        objective,
                        perturbed.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                    )
                    if res2.fun < best_val:
                        best_params = res2.x
                        best_val = res2.fun
                except:
                    pass
    except:
        pass
    
    # Extract final positions
    final_positions = best_params.reshape(N, 3)
    
    # Compute final side length
    final_side = _compute_outer_hex_side(final_positions.tolist())
    
    # Find center of enclosing hexagon (use centroid of inner hexagons as approximation)
    center_x = np.mean(final_positions[:, 0])
    center_y = np.mean(final_positions[:, 1])
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(final_positions[i, 0]), float(final_positions[i, 1]), float(final_positions[i, 2])])
    
    return inner, (float(center_x), float(center_y)), float(final_side), 0.0

# EVOLVE-BLOCK-END
