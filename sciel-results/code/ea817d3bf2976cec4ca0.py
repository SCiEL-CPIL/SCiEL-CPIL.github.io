"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    sqrt3 = math.sqrt(3)
    
    # Axial coordinate to Cartesian for flat-top hexagons
    # For a hex grid where adjacent centers are distance 2 apart (unit hexagons side 1)
    # Flat-top hexagon: vertices at angles 0, 60, 120, 180, 240, 300
    # Inradius (apothem) = sqrt(3)/2, Circumradius = 1
    # Neighbors in axial (q, r) for flat-top:
    # (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)
    # Cartesian: x = 1.5 * q, y = sqrt(3) * (r + q/2.0)
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        return x, y
    
    # Generate hexagonal ring positions
    def hex_ring_positions(ring):
        if ring == 0:
            return [(0, 0)]
        positions = []
        # Start at (ring, 0)
        q, r = ring, 0
        for side in range(6):
            for step in range(ring):
                positions.append((q, r))
                # Directions in axial coordinates for flat-top
                # E: (1, 0), SE: (1, -1), S: (0, -1), W: (-1, 0), NW: (-1, 1), NE: (0, 1)
                dq = [1, 1, 0, -1, -1, 0][side]
                dr = [0, -1, -1, 0, 1, 1][side]
                q += dq
                r += dr
        return positions
    
    # Generate candidate sets of N positions
    # Strategy: Use a compact cluster. For N=11, a hexagon of radius 2 has 19 cells.
    # We can take the first 11 or optimize the selection.
    # Let's generate all positions up to ring 3 and try to find a good subset or just use the densest core.
    
    all_positions = [(0, 0)]
    ring = 1
    while len(all_positions) < N:
        all_positions.extend(hex_ring_positions(ring))
        ring += 1
    
    # The "standard" hexagonal number sequence is 1, 7, 19...
    # For N=11, we take the center (1) + ring 1 (6) + 4 from ring 2.
    # To maximize symmetry and minimize bounding box, we should pick 4 from ring 2 that are symmetric.
    # Ring 2 positions: 12 points.
    # Let's just take the first N for now, but we can also try shifting the set.
    
    base_positions = all_positions[:N]
    
    # Convert to Cartesian
    cartesian_positions = []
    for q, r in base_positions:
        x, y = axial_to_cartesian(q, r)
        cartesian_positions.append((x, y))
    
    # Center the cluster
    cx_mean = sum(p[0] for p in cartesian_positions) / len(cartesian_positions)
    cy_mean = sum(p[1] for p in cartesian_positions) / len(cartesian_positions)
    centered_positions = [(p[0] - cx_mean, p[1] - cy_mean) for p in cartesian_positions]
    
    # We want to find the minimum side length S of a regular hexagon (centered at origin, rotation theta)
    # that contains all these unit hexagons.
    # A unit hexagon at center (x,y) with rotation 0 is contained in the outer hexagon if
    # all its vertices are inside.
    # Vertices of unit hexagon at (x,y):
    # (x + cos(a), y + sin(a)) for a in [0, 60, 120, 180, 240, 300]
    
    # Outer hexagon (center 0,0, side S, rotation theta):
    # Defined by 6 half-planes.
    # Normal vectors for outer hexagon sides (perpendicular to sides, pointing inward):
    # For a flat-top hexagon (vertex at 0), the sides are at angles 30, 90, 150, 210, 270, 330.
    # If we rotate the outer hexagon by theta, the normals are at theta + 30 + k*60.
    # The distance from center to each side is the apothem A = S * sqrt(3) / 2.
    # Condition for point (px, py) to be inside:
    # dot((px, py), n_k) <= A for all k, where n_k is unit normal vector.
    
    # So for each inner hexagon, we need to check its 6 vertices.
    # This is a convex constraint. The minimum S is determined by the maximum "projection"
    # of any vertex onto any normal direction, divided by (sqrt(3)/2).
    
    # Specifically, for a given outer rotation theta:
    # Max over all inner hexagons i, all vertices v of hex i, all outer normals n_k:
    # dot(v, n_k)
    # S_min = (max_projection) / (sqrt(3)/2) = max_projection * 2 / sqrt(3)
    
    # We can optimize theta to minimize S_min.
    # S_min(theta) is the max of a set of sinusoidal functions (projections).
    # It is a convex function? No, max of convex is convex, but projection is linear (sinusoidal).
    # Actually, for a fixed set of points, the function f(theta) = max_k dot(v, n_k(theta))
    # where n_k(theta) = (cos(theta + 30 + k*60), sin(theta + 30 + k*60))
    # is the support function of the set of points in direction n.
    # We want to minimize max_k h_K(n_k(theta)) / (sqrt(3)/2).
    
    # Let's sample theta to find the minimum.
    # Due to 6-fold symmetry of the outer hexagon, we only need to search 0 to 60 degrees.
    
    inner_hex_vertices_angles = [0, 60, 120, 180, 240, 300]
    inner_hex_verts = []
    for ang in inner_hex_vertices_angles:
        rad = math.radians(ang)
        inner_hex_verts.append((math.cos(rad), math.sin(rad)))
    
    # Precompute vertices for each centered inner hexagon
    all_inner_verts = []
    for cx, cy in centered_positions:
        for vx, vy in inner_hex_verts:
            all_inner_verts.append((cx + vx, cy + vy))
    
    # Convert to numpy for speed
    verts_np = np.array(all_inner_verts)
    
    def calc_min_side(theta_deg):
        theta_rad = math.radians(theta_deg)
        # Outer normals
        max_proj = 0.0
        for k in range(6):
            n_ang = theta_rad + math.radians(30 + k * 60)
            nx = math.cos(n_ang)
            ny = math.sin(n_ang)
            # Dot product of all vertices with (nx, ny)
            projs = verts_np[:, 0] * nx + verts_np[:, 1] * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        # S = max_proj / (sqrt(3)/2)
        return max_proj * 2.0 / sqrt3
    
    # Search for optimal theta
    best_theta = 0.0
    best_side = float('inf')
    
    # Coarse search
    for t in np.linspace(0, 60, 601):
        s = calc_min_side(t)
        if s < best_side:
            best_side = s
            best_theta = t
    
    # Fine search around best
    if time.time() < deadline - 5:
        for t in np.linspace(best_theta - 0.5, best_theta + 0.5, 1001):
            if t < 0 or t > 60: continue
            s = calc_min_side(t)
            if s < best_side:
                best_side = s
                best_theta = t
    
    # Construct the result
    # Inner hexagons: centered_positions, angle 0 (flat-top)
    inner = []
    for cx, cy in centered_positions:
        inner.append([cx, cy, 0.0])
    
    # Outer hexagon: center (0,0), side best_side, angle best_theta
    # Note: The "angle" in the return value usually refers to the rotation of the outer hexagon.
    # My calc_min_side used theta as the rotation of the outer hexagon (shifting the normals).
    # Standard flat-top has normals at 30, 90...
    # If I set outer angle to best_theta, the normals are at best_theta + 30, ...
    # This matches my calculation.
    
    return inner, (0.0, 0.0), best_side, best_theta
# EVOLVE-BLOCK-END
