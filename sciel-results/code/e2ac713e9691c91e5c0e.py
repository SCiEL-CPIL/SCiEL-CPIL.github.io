import numpy as np
import time
import itertools

N = 12
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _get_axes(hex_verts):
    """Get the 6 edge normals (axes) of a hexagon."""
    axes = []
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        edge = p2 - p1
        # Normal is perpendicular to edge
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm
        axes.append(normal)
    return np.array(axes)

def _sat_overlap(v1, v2):
    """
    Check if two convex polygons overlap using Separating Axis Theorem.
    Returns True if they overlap (or touch), False if separated.
    """
    axes = np.vstack([_get_axes(v1), _get_axes(v2)])
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False
            
    return True

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check containment: all inner vertices must be inside outer
    for iv in inner_verts:
        for v in iv:
            inside = True
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k+1)%6]
                # Cross product to check side
                # (b-a) x (v-a) should be >= 0 for CCW polygon if v is inside/left
                cross = (b[0]-a[0])*(v[1]-a[1]) - (b[1]-a[1])*(v[0]-a[0])
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False, "outside"
                
    # Check disjointness
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _sat_overlap(inner_verts[i], inner_verts[j]):
                return False, "overlap"
                
    return True, "valid"

def _compute_min_side(inner_hexes, outer_center, outer_angle):
    """
    Compute the minimum side length of the outer hexagon (centered at outer_center, rotated by outer_angle)
    required to contain all inner hexagons.
    """
    # Translate inner hexes relative to outer center
    ox, oy = outer_center
    max_proj = 0.0
    
    # The 6 normals of the outer hexagon
    # Outer hexagon vertices are at angles outer_angle + k*60
    # The normals to the edges are at angles outer_angle + k*60 + 30 (if vertices are at 0,60...)
    # Wait, let's derive properly.
    # A regular hexagon centered at origin with side S has apothem a = S * sqrt(3) / 2.
    # The boundary is defined by n_k . x <= a for k=0..5.
    # The normals n_k point to the midpoints of the edges.
    # If the hexagon is rotated by angle theta, the first edge midpoint is at angle theta.
    # So n_k = [cos(theta + k*pi/3), sin(theta + k*pi/3)].
    
    theta = np.deg2rad(outer_angle)
    for k in range(6):
        angle = theta + k * (np.pi / 3)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        # Project all vertices of all inner hexagons onto this normal
        for h in inner_hexes:
            cx, cy, a_deg = h
            # Translate center to relative to outer center
            rel_cx = cx - ox
            rel_cy = cy - oy
            verts = _hex_vertices((rel_cx, rel_cy), 1.0, a_deg)
            projs = np.dot(verts, normal)
            max_proj = max(max_proj, np.max(projs))
            
    # max_proj is the required apothem
    # S = 2 * apothem / sqrt(3)
    return max_proj * 2.0 / np.sqrt(3.0)

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Strategy: 
    # 1. Generate candidate packings based on hexagonal lattice structures.
    # 2. For each packing, try to "shrink" the outer hexagon by optimizing the outer angle 
    #    and potentially translating/rotating the inner cluster as a rigid body? 
    #    Actually, the inner hexagons can rotate individually. 
    #    However, the densest packing of identical hexagons is the hexagonal lattice.
    #    We will assume the inner hexagons are aligned (angle 0) or rotated by 30 degrees 
    #    to fit better into the outer hexagon corners? 
    #    Actually, if the outer hexagon has a "flat" side on top, and inner hexagons are 
    #    "pointy" top, they fit differently.
    
    # Let's generate specific lattice configurations.
    # The "center + 6 neighbors" is 7 hexagons.
    # We need 12. 
    # Config A: Center (1) + Ring 1 (6) + Part of Ring 2 (5).
    # Ring 2 positions in hex lattice (distance 2 from center):
    # (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1)
    # We can pick 5 of these to minimize the bounding box.
    
    # Hex lattice coordinates to Cartesian:
    # x = 1.5 * q
    # y = sqrt(3) / 2 * q + sqrt(3) * r
    # This is for pointy-top? No, standard axial coordinates.
    # Let's use a simpler mapping.
    # Distance between centers of touching hexagons is sqrt(3).
    # Let's define centers in Cartesian directly.
    
    def make_hex_packing(config_type):
        centers = []
        angles = []
        
        if config_type == 'compact_12':
            # Center
            centers.append((0.0, 0.0))
            angles.append(0.0)
            
            # Ring 1: 6 hexagons
            d = np.sqrt(3)
            for k in range(6):
                a = np.pi/3 * k
                centers.append((d*np.cos(a), d*np.sin(a)))
                angles.append(0.0)
            
            # Ring 2: Pick 5 positions to be compact
            # Full ring 2 has 12 positions. We need 5 more.
            # To minimize bounding box, we should pick positions that are "inside" the convex hull 
            # of the outer hexagon as much as possible.
            # The outer hexagon will likely be oriented to match the lattice symmetry.
            
            # Let's try picking Ring 2 positions at angles: 0, 60, 120, 180, 240 (every other one)
            # Or 30, 90, 150, 210, 270?
            
            # Let's generate all 12 ring 2 positions and pick a subset of 5 that minimizes 
            # the max distance from origin? No, we need to minimize the outer hexagon size.
            
            # Let's just try a few specific subsets of Ring 2.
            # Subset 1: Angles 0, 60, 120, 180, 240 (aligned with Ring 1)
            # Subset 2: Angles 30, 90, 150, 210, 270 (offset)
            
            # Let's return a list of possible inner hexagons for this config type
            pass
            
        # Instead of hardcoding, let's generate a few specific valid packings manually.
        
        packings = []
        
        # Packing 1: Center + 6 + 5 (aligned)
        c1 = []
        a1 = []
        c1.append((0,0)); a1.append(0)
        d = np.sqrt(3)
        for k in range(6):
            ang = k * np.pi / 3
            c1.append((d*np.cos(ang), d*np.sin(ang)))
            a1.append(0)
        # Add 5 from ring 2 (dist 2*sqrt(3))
        d2 = 2 * np.sqrt(3)
        # Pick indices 0, 1, 2, 3, 4 of the 12 ring 2 spots? 
        # Ring 2 spots are at angles 0, 30, 60... ? No.
        # In hex lattice, ring 2 has 12 vertices.
        # Let's just place 5 more hexagons at specific positions that are valid.
        # (2*sqrt(3), 0), (sqrt(3), 3), (0, 2*sqrt(3)), (-sqrt(3), 3), (-2*sqrt(3), 0)
        # Check distance to center: 2*sqrt(3) ~ 3.46.
        # Check distance to neighbors in Ring 1:
        # Neighbor at (sqrt(3), 0). Dist to (2*sqrt(3), 0) is sqrt(3). OK.
        # Neighbor at (sqrt(3)/2, 1.5). Dist to (sqrt(3), 3)?
        # dx = sqrt(3)/2, dy = 1.5. dist^2 = 3/4 + 2.25 = 3. dist = sqrt(3). OK.
        
        pos_ring2 = [
            (2*np.sqrt(3), 0),
            (np.sqrt(3), 3),
            (0, 2*np.sqrt(3)),
            (-np.sqrt(3), 3),
            (-2*np.sqrt(3), 0)
        ]
        for p in pos_ring2:
            c1.append(p)
            a1.append(0)
            
        packings.append(list(zip(c1, a1)))
        
        # Packing 2: Center + 6 + 5 (offset ring 2)
        c2 = []
        a2 = []
        c2.append((0,0)); a2.append(0)
        for k in range(6):
            ang = k * np.pi / 3
            c2.append((d*np.cos(ang), d*np.sin(ang)))
            a2.append(0)
        
        # Offset ring 2 positions
        # These are the "gaps" between the aligned ring 2 positions.
        # Angle 30 deg: (2*sqrt(3)*cos(30), 2*sqrt(3)*sin(30)) = (3, sqrt(3))
        # Angle 90 deg: (0, 2*sqrt(3)) -- wait, this is same as above?
        # Let's re-evaluate hex lattice geometry.
        # Center (0,0).
        # Ring 1: (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
        # Ring 2:
        # (2*sqrt(3), 0)
        # (3, sqrt(3))
        # (1.5, 3)
        # (-1.5, 3)
        # (-3, sqrt(3))
        # (-2*sqrt(3), 0)
        # (-3, -sqrt(3))
        # (-1.5, -3)
        # (1.5, -3)
        # (3, -sqrt(3))
        # (0, 2*sqrt(3)) -- Wait, is (0, 2*sqrt(3)) a valid ring 2 position?
        # Distance from (0,0) is 2*sqrt(3) ~ 3.46.
        # Distance from (sqrt(3)/2, 1.5) to (0, 2*sqrt(3))?
        # dx = -sqrt(3)/2, dy = 2*sqrt(3) - 1.5.
        # 2*1.732 = 3.464. 3.464 - 1.5 = 1.964.
        # dist^2 = 3/4 + 1.964^2 = 0.75 + 3.85 = 4.6. sqrt(4.6) = 2.14.
        # This is > sqrt(3) (1.732). So they don't touch.
        # Are they disjoint? Yes.
        # But is (0, 2*sqrt(3)) a lattice point?
        # In a hex lattice, the points at distance 2*sqrt(3) from center are 12 points.
        # Let's just use the 5 points that form a pentagon-like shape to minimize max radius.
        
        pos_ring2_offset = [
            (3, np.sqrt(3)),
            (1.5, 3),
            (-1.5, 3),
            (-3, np.sqrt(3)),
            (0, -2*np.sqrt(3)) # Pick one from bottom to balance?
        ]
        # Actually, just picking 5 out of 12 arbitrarily might be bad.
        # Let's try to pick the 5 that minimize the max projection on 6 axes.
        # This is a small optimization problem.
        
        # Let's just use the first packing for now and optimize the outer hexagon orientation.
        
        for packing in packings:
            # Try different outer angles
            # The packing has 6-fold symmetry (mostly).
            # Let's try outer angles 0, 30.
            for outer_angle in [0.0, 30.0]:
                # Calculate min side
                side = _compute_min_side(packing, (0,0), outer_angle)
                
                # Verify validity
                is_valid, _ = _check_validity(packing, side, (0,0), outer_angle)
                
                if is_valid and side < best_side:
                    best_side = side
                    best_inner = [list(p) for p in packing]
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = outer_angle
                    
                if time.time() - start_time > TIME_LIMIT - 10:
                    return (best_inner, best_outer_center, best_side, best_outer_angle)
            
        # Try a second packing: 2 rows of 6?
        # No, hexagons don't stack in rows efficiently without gaps.
        # Try a "flower" with 11 in ring 1 and 1 in center?
        # Ring 1 has 6 spots. Can't fit 11.
        
        # Try a packing with rotated inner hexagons?
        # If we rotate inner hexagons by 30 degrees, they might fit into the corners of the outer hexagon better?
        # Let's try packing 1 but with inner angles = 30.
        packing_rot = [(c, 30.0) for c, _ in packings[0]]
        for outer_angle in [0.0, 30.0]:
            side = _compute_min_side(packing_rot, (0,0), outer_angle)
            is_valid, _ = _check_validity(packing_rot, side, (0,0), outer_angle)
            if is_valid and side < best_side:
                best_side = side
                best_inner = [list(p) for p in packing_rot]
                best_outer_center = (0.0, 0.0)
                best_outer_angle = outer_angle
                
    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
