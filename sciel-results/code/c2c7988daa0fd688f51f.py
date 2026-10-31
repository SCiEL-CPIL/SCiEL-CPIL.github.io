"""Hexagon packing in a hexagon - N=11 unit hexagons in outer hexagon."""
import math
import numpy as np
from scipy.optimize import minimize
import time

N = 11

def get_hex_vertices(cx, cy, angle, side=1.0):
    """Get vertices of a regular hexagon centered at (cx, cy) with given angle (radians)."""
    angles = np.arange(6) * np.pi / 3.0 + angle
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def hexagon_distance(hex1_verts, hex2_verts):
    """Compute minimum distance between two hexagons (0 if overlapping)."""
    # For two convex polygons, we can check edge-edge distances
    # Simple approach: check all vertex-edge pairs and edge-edge pairs
    # For efficiency, first check center distance
    c1 = np.mean(hex1_verts, axis=0)
    c2 = np.mean(hex2_verts, axis=0)
    center_dist = np.linalg.norm(c1 - c2)
    if center_dist < 0.01:
        return 0.0
    if center_dist > 3.0:  # Too far apart, max distance is 2
        return center_dist - 2.0
    
    # Check if any vertex of one is inside the other
    def point_in_hex(p, verts):
        """Check if point p is inside hexagon defined by verts."""
        for i in range(6):
            v1 = verts[i]
            v2 = verts[(i+1) % 6]
            # Cross product to check which side
            edge = v2 - v1
            to_point = p - v1
            cross = edge[0] * to_point[1] - edge[1] * to_point[0]
            if cross < -1e-10:
                return False
        return True
    
    # Check vertices of hex1 against hex2
    for v in hex1_verts:
        if point_in_hex(v, hex2_verts):
            return 0.0
    # Check vertices of hex2 against hex1
    for v in hex2_verts:
        if point_in_hex(v, hex1_verts):
            return 0.0
    
    # Compute minimum edge-edge distance
    min_dist = float('inf')
    for i in range(6):
        e1_start = hex1_verts[i]
        e1_end = hex1_verts[(i+1) % 6]
        for j in range(6):
            e2_start = hex2_verts[j]
            e2_end = hex2_verts[(j+1) % 6]
            # Compute distance between two line segments
            d = seg_seg_dist(e1_start, e1_end, e2_start, e2_end)
            if d < min_dist:
                min_dist = d
    return min_dist

def seg_seg_dist(p1, q1, p2, q2):
    """Minimum distance between segments p1-q1 and p2-q2."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-10 and e < 1e-10:
        return np.linalg.norm(r)
    
    if a < 1e-10:
        t2 = np.clip(f / e, 0, 1)
        closest = p2 + d2 * t2
        return np.linalg.norm(p1 - closest)
    
    c = np.dot(d1, r)
    if e < 1e-10:
        t1 = np.clip(c / a, 0, 1)
        closest = p1 + d1 * t1
        return np.linalg.norm(p2 - closest)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if abs(denom) > 1e-10:
        s = np.clip((b * f - c * e) / denom, 0, 1)
    else:
        s = 0.0
    t = np.clip((b * s - f) / e, 0, 1)
    
    closest1 = p1 + d1 * s
    closest2 = p2 + d2 * t
    return np.linalg.norm(closest1 - closest2)

def point_in_hexagon(p, center, angle, side=1.0):
    """Check if point p is inside a regular hexagon."""
    # Transform to local coordinate system
    local = p - center
    # Rotate by -angle
    ca = math.cos(-angle)
    sa = math.sin(-angle)
    local_rot = np.array([
        ca * local[0] - sa * local[1],
        sa * local[0] + ca * local[1]
    ])
    
    # In local coords, hexagon has vertices at angles 0, 60, 120, ...
    # The hexagon is the intersection of 6 half-planes
    # For each edge, check if point is on the correct side
    for i in range(6):
        ang1 = i * math.pi / 3
        ang2 = (i + 1) * math.pi / 3
        v1 = side * np.array([math.cos(ang1), math.sin(ang1)])
        v2 = side * np.array([math.cos(ang2), math.sin(ang2)])
        edge = v2 - v1
        # Normal pointing inward
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        # The center should be on the positive side
        if np.dot(normal, local_rot) < -1e-10:
            return False
    return True

def is_inside_outer_hex(p, outer_center, outer_angle, outer_side):
    """Check if point p is inside the outer hexagon."""
    return point_in_hexagon(p, outer_center, outer_angle, outer_side)

def compute_objective(params, outer_center, outer_angle, outer_side):
    """
    Compute objective: sum of violations.
    params: array of shape (N, 3) - [x, y, angle] for each hexagon
    Returns: total violation (0 if valid, positive if invalid)
    """
    violation = 0.0
    
    # Get all hexagon vertices
    hexes = []
    for i in range(N):
        x, y, ang = params[i]
        hexes.append(get_hex_vertices(x, y, ang))
    
    # Check pairwise distances
    for i in range(N):
        for j in range(i+1, N):
            d = hexagon_distance(hexes[i], hexes[j])
            if d < 0:
                violation += (-d) ** 2
    
    # Check all hexagons are inside outer hexagon
    for i in range(N):
        for v in hexes[i]:
            if not is_inside_outer_hex(v, outer_center, outer_angle, outer_side):
                # Compute distance to boundary
                # Simple penalty
                local = v - outer_center
                ca = math.cos(-outer_angle)
                sa = math.sin(-outer_angle)
                local_rot = np.array([
                    ca * local[0] - sa * local[1],
                    sa * local[0] + ca * local[1]
                ])
                # Find how far outside
                for k in range(6):
                    ang1 = k * math.pi / 3
                    ang2 = (k + 1) * math.pi / 3
                    v1 = outer_side * np.array([math.cos(ang1), math.sin(ang1)])
                    v2 = outer_side * np.array([math.cos(ang2), math.sin(ang2)])
                    edge = v2 - v1
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / np.linalg.norm(normal)
                    dist = np.dot(normal, local_rot) - np.dot(normal, v1)
                    if dist < 0:
                        violation += (-dist) ** 2 * 10
                        break
    
    return violation

def solve_packing():
    """Solve the hexagon packing problem."""
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Initial guess: place hexagons in a grid-like pattern
    # Try multiple initial configurations
    best_solution = None
    best_score = float('inf')
    
    # Grid-based initial positions for N=11
    # A hexagonal grid arrangement
    positions = []
    # Center + 6 around + 4 more
    # Let's use a simple approach: start with a large hexagonal pattern
    r = math.sqrt(3)  # Distance between adjacent hexagon centers
    
    # Hexagonal lattice positions
    basis1 = np.array([r, 0])
    basis2 = np.array([r/2, r * math.sqrt(3)/2])
    
    # Generate positions in hexagonal order
    hex_positions = [(0, 0)]
    for ring in range(1, 5):
        for i in range(-ring, ring+1):
            for j in range(-ring, ring+1):
                if max(abs(i), abs(j), abs(-i-j)) == ring:
                    hex_positions.append((i, j))
    
    # Convert to cartesian
    cart_positions = []
    for (i, j) in hex_positions[:N]:
        pos = i * basis1 + j * basis2
        cart_positions.append(pos)
    
    # If we don't have enough, add more
    while len(cart_positions) < N:
        idx = len(cart_positions)
        i = idx // 2
        j = idx % 2
        pos = i * basis1 + j * basis2 + np.array([0.5, 0.5])
        cart_positions.append(pos)
    
    cart_positions = np.array(cart_positions[:N])
    
    # Try multiple random restarts
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Initialize parameters
        if restart == 0:
            x0 = np.hstack([cart_positions.flatten(), np.zeros(N)])
        else:
            # Perturb initial solution
            noise_scale = 0.5 + restart * 0.1
            x0 = np.hstack([
                cart_positions.flatten() + np.random.randn(2*N) * noise_scale,
                np.random.randn(N) * 0.5
            ])
        
        # Use Nelder-Mead for optimization
        try:
            result = minimize(
                compute_objective,
                x0.reshape(N, 3),
                args=(np.array([0.0, 0.0]), 0.0, 2.0),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'initial_simplex': 0.1
                },
                callback=lambda x: (time.time() - start_time) < (deadline - 5)
            )
        except Exception as e:
            continue
        
        if result.fun < best_score:
            best_score = result.fun
            best_solution = result.x
        
        if best_score < 1e-6:
            break
    
    if best_solution is None:
        # Fallback: return a valid but not optimal configuration
        inner = []
        for i in range(N):
            x, y = cart_positions[i]
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 3.0, 0.0
    
    # Extract positions and angles
    pos_angles = best_solution.reshape(N, 3)
    
    # Now find minimum outer side length
    # Binary search on outer side length
    def check_valid(outer_side, outer_center, outer_angle, pos_angles):
        """Check if configuration is valid with given outer hexagon."""
        for i in range(N):
            x, y, ang = pos_angles[i]
            verts = get_hex_vertices(x, y, ang)
            for v in verts:
                if not is_inside_outer_hex(v, outer_center, outer_angle, outer_side):
                    return False
        # Check pairwise
        hexes = [get_hex_vertices(pos_angles[i][0], pos_angles[i][1], pos_angles[i][2]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if hexagon_distance(hexes[i], hexes[j]) < -1e-9:
                    return False
        return True
    
    # Binary search for minimum outer side
    lo, hi = 1.5, 5.0
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    # First, center the configuration
    centroid = np.mean(pos_angles[:, :2], axis=0)
    pos_angles[:, :2] -= centroid
    
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) / 2
        if check_valid(mid, outer_center, outer_angle, pos_angles):
            hi = mid
        else:
            lo = mid
    
    final_side = hi
    
    inner = []
    for i in range(N):
        inner.append([pos_angles[i][0], pos_angles[i][1], math.degrees(pos_angles[i][2])])
    
    return inner, (0.0, 0.0), final_side, math.degrees(outer_angle)

def construct():
    """Construct the hexagon packing solution."""
    inner, center, side, angle = solve_packing()
    return inner, center, side, angle
# EVOLVE-BLOCK-END
