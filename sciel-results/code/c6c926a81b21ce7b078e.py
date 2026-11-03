"""Unit regular hexagons inside a regular hexagon - Honeycomb Defect approach."""
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
import math

N = 11
start_time = time.time()
DEADLINE = 110.0  # Safety margin before 120s limit

# Hexagon geometry constants
SQRT3 = math.sqrt(3)

# Regular hexagon of side s=1:
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
# Distance from center to vertex = 1 (circumradius = side length for regular hexagon)
# Distance from center to edge midpoint = sqrt(3)/2 (apothem)

def hexagon_vertices(center, side, rotation_deg):
    """Return vertices of a regular hexagon with given center, side length, rotation."""
    cx, cy = center
    rot = math.radians(rotation_deg)
    verts = []
    for i in range(6):
        angle = rot + i * math.pi / 3
        x = cx + side * math.cos(angle)
        y = cy + side * math.sin(angle)
        verts.append([x, y])
    return np.array(verts)

def point_in_hexagon(px, py, center, side, rotation_deg):
    """Check if point (px, py) is inside a regular hexagon."""
    cx, cy = center
    rot = math.radians(rotation_deg)
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    # Rotate back by -rotation
    x_local = dx * math.cos(-rot) - dy * math.sin(-rot)
    y_local = dx * math.sin(-rot) + dy * math.cos(-rot)
    
    # Check if inside hexagon using the 6 half-plane inequalities
    # For a hexagon centered at origin with circumradius = side:
    # The hexagon is the intersection of 6 half-planes
    # At angles 0, 60, 120, 180, 240, 300, the supporting lines are at distance apothem = side*sqrt(3)/2
    apothem = side * SQRT3 / 2
    for i in range(6):
        angle = i * math.pi / 3
        # Normal direction points outward
        nx = math.cos(angle)
        ny = math.sin(angle)
        # Project point onto normal
        proj = x_local * nx + y_local * ny
        if proj > apothem + 1e-10:
            return False
    return True

def hexagon_overlap(h1_center, h1_side, h1_rot, h2_center, h2_side, h2_rot):
    """Check if two hexagons overlap (interiors intersect)."""
    # Simple check: distance between centers vs sum of circumradii
    dx = h1_center[0] - h2_center[0]
    dy = h1_center[1] - h2_center[1]
    dist = math.sqrt(dx * dx + dy * dy)
    
    # For unit hexagons, minimum center distance for non-overlap depends on orientation
    # Conservative bound: if dist < sum of apothems, they might overlap
    # More precise: check if any vertex of one is inside the other, or vice versa
    apothem1 = h1_side * SQRT3 / 2
    apothem2 = h2_side * SQRT3 / 2
    
    # Quick reject
    if dist > h1_side + h2_side:
        return False
    
    # Check if any vertex of hex1 is inside hex2
    verts1 = hexagon_vertices(h1_center, h1_side, h1_rot)
    for v in verts1:
        if point_in_hexagon(v[0], v[1], h2_center, h2_side, h2_rot):
            return True
    
    # Check if any vertex of hex2 is inside hex1
    verts2 = hexagon_vertices(h2_center, h2_side, h2_rot)
    for v in verts2:
        if point_in_hexagon(v[0], v[1], h1_center, h1_side, h1_rot):
            return True
    
    # Check edge-edge intersections (simplified: if centers are close, assume overlap)
    if dist < 0.01:
        return True
    
    return False

def compute_enclosing_side(hex_positions, hex_angles, outer_center, outer_rotation):
    """Compute the minimum side length of an outer hexagon (with given center and rotation)
    that contains all inner hexagons.
    
    hex_positions: list of (x, y) centers of inner unit hexagons
    hex_angles: list of rotation angles in degrees
    outer_center: (x, y) of outer hexagon center
    outer_rotation: rotation of outer hexagon in degrees
    """
    ocx, ocy = outer_center
    orot = math.radians(outer_rotation)
    
    max_dist = 0.0
    
    for (hx, hy), hrot_deg in zip(hex_positions, hex_angles):
        hrot = math.radians(hrot_deg)
        # For each inner hexagon, find the maximum distance from outer center
        # to any point of the inner hexagon, measured in the direction that maximizes
        # the support function of the outer hexagon
        
        # The support function of a regular hexagon in direction theta is:
        # h(theta) = apothem / cos((theta - nearest_vertex_angle) mod 60)
        # But for containment, we need: for every direction, the projection of all points
        # of inner hexagons from outer center must be within the support function of outer hexagon
        
        # Simpler approach: the outer hexagon contains a set of points iff for each of the 6
        # supporting half-planes, all points satisfy the inequality.
        
        # The 6 supporting lines of the outer hexagon (centered at outer_center, rotated by outer_rotation):
        # At angles outer_rotation + k*60 degrees, the line is at distance R_outer * sqrt(3)/2 from center
        # where R_outer is the circumradius = side length
        
        # So we need: for all 6 directions k=0..5, max over all points p of (p - outer_center) . n_k <= side * sqrt(3)/2
        # where n_k is the unit normal at angle outer_rotation + k*pi/3
        
        # For each inner hexagon, the maximum projection in direction n is:
        # max over vertices of (vertex - outer_center) . n
        
        # Actually, the support function of a convex polygon in direction n is max over vertices of (v - c) . n
        # where c is the polygon center... no, it's max over vertices of v . n - c . n... 
        
        # Let me think more carefully. The support function h_P(n) of a set P in direction n is max_{p in P} p . n
        # For a hexagon centered at c with vertices v_i = c + r_i, h_P(n) = c . n + max_i (r_i . n)
        
        # The outer hexagon contains all inner hexagons iff for all 6 directions n_k:
        # max over all inner hexagons of h_inner(n_k) <= h_outer(n_k)
        # where h_outer(n_k) = outer_center . n_k + R_outer * sqrt(3)/2  (since the outer hexagon's support in direction n_k is at distance R_outer*sqrt(3)/2 from center in that direction)
        
        # Wait, the support function of the outer hexagon in direction n_k (where n_k is one of the 6 normal directions) is:
        # h_outer(n_k) = outer_center . n_k + R_outer * sqrt(3)/2
        
        # And we need: for each k, max over all points p in union of inner hexagons of p . n_k <= h_outer(n_k)
        
        # So: max over inner hexagons and vertices of (v . n_k) <= outer_center . n_k + R_outer * sqrt(3)/2
        # => R_outer >= [max over all v (v . n_k) - outer_center . n_k] / (sqrt(3)/2)
        
        # So the required side length is:
        # R_outer = max over k of [max over all inner hex vertices v of (v . n_k) - outer_center . n_k] / (sqrt(3)/2)
        
        for i in range(6):
            angle_n = orot + i * math.pi / 3
            nx = math.cos(angle_n)
            ny = math.sin(angle_n)
            
            # Get vertices of this inner hexagon
            verts = hexagon_vertices((hx, hy), 1.0, hrot_deg)
            for v in verts:
                proj = v[0] * nx + v[1] * ny
                oc_proj = ocx * nx + ocy * ny
                required_R = (proj - oc_proj) / (SQRT3 / 2)
                if required_R > max_dist:
                    max_dist = required_R
    
    return max_dist

def solve_minimum_side(hex_positions, hex_angles):
    """Find the minimum side length outer hexagon that contains all inner hexagons,
    optimizing over outer center and rotation."""
    
    # The optimal outer hexagon will be tightly enclosing the cluster.
    # We can search over rotation and center.
    
    # For a given rotation, the optimal center is determined by the constraints.
    # Let's do a simple grid search over rotation (0 to 60 degrees) and center.
    
    # First, compute the bounding box of all hexagon vertices to estimate center
    all_verts = []
    for (hx, hy), hrot_deg in zip(hex_positions, hex_angles):
        verts = hexagon_vertices((hx, hy), 1.0, hrot_deg)
        all_verts.extend(verts.tolist())
    
    all_verts = np.array(all_verts)
    center_x = (all_verts[:, 0].min() + all_verts[:, 0].max()) / 2
    center_y = (all_verts[:, 1].min() + all_verts[:, 1].max()) / 2
    
    best_side = float('inf')
    
    # Search over rotation
    for rot_deg in np.linspace(0, 60, 121):
        # For each rotation, optimize center
        # The center that minimizes the required side is near the centroid of the cluster
        # Let's try a small grid around the estimated center
        
        # Compute the "center" that balances the extents in all 6 directions
        # For a given rotation, we want to choose (cx, cy) to minimize:
        # max over k of [max_v (v . n_k) - (cx*nx_k + cy*ny_k)] / (sqrt(3)/2)
        
        # This is a linear programming problem in (cx, cy) for fixed rotation.
        # Let's solve it approximately.
        
        # For each direction k, the constraint is:
        # cx*nx_k + cy*ny_k >= max_v (v . n_k) - R * sqrt(3)/2
        # where R is the side length we want to minimize.
        
        # So we want to maximize cx*nx_k + cy*ny_k - R*sqrt(3)/2 >= max_v(v . n_k) for all k
        # Equivalent to: R >= [max_v(v . n_k) - cx*nx_k - cy*ny_k] / (sqrt(3)/2) for all k
        # Minimize R over (cx, cy).
        
        # This is: minimize max_k [a_k - cx*nx_k - cy*ny_k] where a_k = max_v(v . n_k)
        # = minimize max_k [a_k - d_k . (cx, cy)] where d_k = (nx_k, ny_k)
        
        # This is a convex optimization problem. Let's use a simple approach:
        # The optimal (cx, cy) makes all 6 constraints tight if possible, or some subset.
        
        # Simple approach: try a grid of centers around the centroid
        
        cx_range = np.linspace(center_x - 3, center_x + 3, 15)
        cy_range = np.linspace(center_y - 3, center_y + 3, 15)
        
        for cx in cx_range:
            for cy in cy_range:
                side = compute_enclosing_side(hex_positions, hex_angles, (cx, cy), rot_deg)
                if side < best_side:
                    best_side = side
    
    return best_side

def construct():
    """Construct 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
    global start_time, DEADLINE
    
    # Honeycomb lattice positions for 14 hexagons (ring of 12 + center + 1 extra)
    # Standard honeycomb: positions are integer combinations of basis vectors
    # e1 = (sqrt(3), 0), e2 = (sqrt(3)/2, 3/2)
    
    e1 = np.array([SQRT3, 0.0])
    e2 = np.array([SQRT3/2, 1.5])
    
    # 14-hexagon cluster: center + 6 ring + 7 more? 
    # Actually, a natural 14-hex cluster: hex numbers up to some ring
    # Ring 0: 1 hex (center)
    # Ring 1: 6 hexes
    # Ring 2: 12 hexes
    # Total: 1 + 6 + 12 = 19 for two rings
    
    # Let's use a 14-hex cluster: center + ring 1 (6) + 7 from ring 2
    # Or: a 4x4 diamond shape in hex lattice
    
    # Simpler: use a 14-hex "blob" and remove 3
    # Let's define positions in hex lattice coordinates (i, j)
    # Convert to cartesian: pos = i*e1 + j*e2
    
    # 14-hex cluster: let's use a roughly circular shape
    # Coordinates (i, j) where i+j, i-j, etc. are bounded
    
    # Hex lattice neighbors: (i+1,j), (i-1,j), (i,j+1), (i,j-1), (i+1,j-1), (i-1,j+1)
    
    # 14-hex cluster: center (0,0), ring 1: 6 hexes, then 7 from ring 2
    ring1 = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    ring2_partial = [(2,0), (-2,0), (0,2), (0,-2), (2,-2), (-2,2), (1,1)]
    
    # Total: 1 + 6 + 7 = 14
    all_14 = [(0,0)] + ring1 + ring2_partial
    
    # Now enumerate ways to remove 3 hexagons from these 14 to get 11
    # There are C(14,3) = 364 combinations
    
    best_result = None
    best_side = float('inf')
    
    for removal in itertools.combinations(range(14), 3):
        if time.time() - start_time > DEADLINE:
            break
        
        removed_set = set(removal)
        selected = [all_14[i] for i in range(14) if i not in removed_set]
        
        if len(selected) != 11:
            continue
        
        # Convert to cartesian
        hex_positions = []
        for (i, j) in selected:
            pos = i * e1 + j * e2
            hex_positions.append((pos[0], pos[1]))
        
        # All inner hexagons have rotation 0 (aligned with outer)
        hex_angles = [0.0] * 11
        
        # Check for overlaps (they shouldn't overlap in honeycomb lattice)
        # Verify no overlaps
        has_overlap = False
        for a in range(11):
            for b in range(a + 1, 11):
                if hexagon_overlap(hex_positions[a], 1.0, 0.0, hex_positions[b], 1.0, 0.0):
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if has_overlap:
            continue
        
        # Compute minimum enclosing side
        side = solve_minimum_side(hex_positions, hex_angles)
        
        if side < best_side:
            best_side = side
            # Find optimal center and rotation
            # We need to track the best center and rotation
            # Let's recompute to get them
            all_verts = []
            for (hx, hy) in hex_positions:
                verts = hexagon_vertices((hx, hy), 1.0, 0.0)
                all_verts.extend(verts.tolist())
            all_verts = np.array(all_verts)
            
            # Do a finer search for the best center and rotation
            cx0 = (all_verts[:, 0].min() + all_verts[:, 0].max()) / 2
            cy0 = (all_verts[:, 1].min() + all_verts[:, 1].max()) / 2
            
            best_local_side = float('inf')
            best_cx = cx0
            best_cy = cy0
            best_rot = 0.0
            
            for rot_deg in np.linspace(0, 60, 61):
                for cx in np.linspace(cx0 - 2, cx0 + 2, 11):
                    for cy in np.linspace(cy0 - 2, cy0 + 2, 11):
                        s = compute_enclosing_side(hex_positions, hex_angles, (cx, cy), rot_deg)
                        if s < best_local_side:
                            best_local_side = s
                            best_cx = cx
                            best_cy = cy
                            best_rot = rot_deg
            
            if best_local_side < best_side:
                best_side = best_local_side
                best_result = {
                    'positions': hex_positions,
                    'angles': hex_angles,
                    'center': (best_cx, best_cy),
                    'side': best_side,
                    'outer_rotation': best_rot
                }
    
    if time.time() - start_time > DEADLINE:
        if best_result is None:
            # Fallback: linear arrangement
            inner = [[3.0 * k, 0.0, 0.0] for k in range(11)]
            return inner, (0.0, 0.0), 40.0, 0.0
        else:
            inner = [[p[0], p[1], a] for p, a in zip(best_result['positions'], best_result['angles'])]
            return inner, best_result['center'], best_result['side'], best_result['outer_rotation']
    
    if best_result is None:
        # Fallback
        inner = [[3.0 * k, 0.0, 0.0] for k in range(11)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = [[p[0], p[1], a] for p, a in zip(best_result['positions'], best_result['angles'])]
    return inner, best_result['center'], best_result['side'], best_result['outer_rotation']
