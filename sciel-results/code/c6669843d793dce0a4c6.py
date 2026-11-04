"""Packing 11 unit hexagons inside a regular hexagon; minimize outer side length."""
import numpy as np
import time

N = 11
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
    a = np.deg2rad(angle_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    # Unit hexagon vertices (side 1) at angles 0,60,...,300
    base = np.array([[np.cos(np.deg2rad(d)), np.sin(np.deg2rad(d))] for d in range(0, 360, 60)])
    # Apply rotation
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    verts = base @ R.T * side + np.array([cx, cy])
    return verts

def _hexagon_halfplanes(verts):
    """Return (n, 2) inward normals and (n,) offsets for hexagon with given verts (CCW order)."""
    n = len(verts)
    normals = np.zeros((n, 2))
    offsets = np.zeros(n)
    for i in range(n):
        v0 = verts[i]
        v1 = verts[(i + 1) % n]
        edge = v1 - v0
        # Inward normal for CCW polygon
        norm = np.array([edge[1], -edge[0]])
        norm /= np.linalg.norm(norm)
        normals[i] = norm
        offsets[i] = norm @ v0
    return normals, offsets

def _point_in_hexagon(point, normals, offsets):
    """Check if point is inside all halfplanes (normals·x >= offsets)."""
    return np.all(normals @ point >= offsets - 1e-10)

def _hexagon_inside_hexagon(inner_verts, outer_normals, outer_offsets):
    """Check if all inner vertices are inside the outer hexagon (halfplane test)."""
    return np.all(outer_normals @ inner_verts.T >= outer_offsets[:, None] - 1e-9)

def _min_outer_side_for_packing(inner_positions, inner_angles_deg, outer_center, outer_angle_deg):
    """
    Given fixed inner hexagons and outer hexagon center/angle, find minimum outer side length.
    We compute the maximum distance from outer center to any inner vertex in the rotated frame,
    then find the smallest hexagon side that contains all points.
    
    Actually, for a regular hexagon with given center and rotation, the constraint is:
    For each of the 6 halfplanes, max over all inner vertices of (normal_i · (v - center)) <= s
    So minimum s = max over i, over all vertices v of (normal_i · (v - center))
    """
    a = np.deg2rad(outer_angle_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # Unit hexagon normals (for side 1) at the 6 edges
    base = np.array([[np.cos(np.deg2rad(d)), np.sin(np.deg2rad(d))] for d in range(0, 360, 60)])
    Rb = base @ R.T
    # Halfplane normals for the unit hexagon
    n = len(Rb)
    normals = np.zeros((n, 2))
    for i in range(n):
        v0 = Rb[i]
        v1 = Rb[(i + 1) % n]
        edge = v1 - v0
        norm = np.array([edge[1], -edge[0]])
        norm /= np.linalg.norm(norm)
        normals[i] = norm
    
    max_s = 0.0
    for pos, ang in zip(inner_positions, inner_angles_deg):
        inner_verts = _hex_vertices(pos[0], pos[1], 1.0, ang)
        # Shift by outer center
        shifted = inner_verts - outer_center
        # Project onto each normal
        for i in range(n):
            proj = shifted @ normals[i]
            max_s = max(max_s, np.max(proj))
    return max_s

def _evaluate_packing(inner_positions, inner_angles_deg, outer_center, outer_angle_deg):
    """Return (side_length, is_valid)."""
    side = _min_outer_side_for_packing(inner_positions, inner_angles_deg, outer_center, outer_angle_deg)
    # Verify validity: check all inner vertices are inside
    a = np.deg2rad(outer_angle_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    base = np.array([[np.cos(np.deg2rad(d)), np.sin(np.deg2rad(d))] for d in range(0, 360, 60)])
    Rb = base @ R.T
    n = len(Rb)
    normals = np.zeros((n, 2))
    for i in range(n):
        v0 = Rb[i]
        v1 = Rb[(i + 1) % n]
        edge = v1 - v0
        norm = np.array([edge[1], -edge[0]])
        norm /= np.linalg.norm(norm)
        normals[i] = norm
    offsets = normals @ outer_center
    
    for pos, ang in zip(inner_positions, inner_angles_deg):
        inner_verts = _hex_vertices(pos[0], pos[1], 1.0, ang)
        if not np.all(normals @ inner_verts.T >= offsets[:, None] - 1e-7):
            return side, False
    return side, True

def _build_hex_lattice_positions(radius_cells=2, angle_step=30.0):
    """
    Generate candidate positions for 11 hexagons on a hexagonal lattice.
    Use a compact cluster of 11 hexagons: center + 6 around + 4 more, or a 11-cell hex pattern.
    """
    # Hex lattice: positions in axial coordinates
    # A compact 11-hexagon cluster: a hexagon of radius 2 has 1+6+12=19, too many.
    # Use a cross/plus shape or a small cluster.
    # Let's use: center + 6 neighbors + 4 of the 12 second-ring = 11
    # Or simpler: a 3x3 diamond in hex lattice minus corners... 
    # Actually, for 11 hexagons, a natural shape is a hexagon of "radius" ~1.8
    # Let's just place them in a hex lattice pattern:
    # Axial coords: (q, r) -> position: x = (q + r/2) * sqrt(3) * s, y = r * 1.5 * s
    # where s is the side length (=1 for unit hexagons). But we want them disjoint,
    # so center-to-center distance must be >= 2 (for side-1 hexagons, min distance is 2*apothem = sqrt(3)*2? No.
    # For two unit hexagons side by side sharing an edge, center distance = 2 * apothem = 2*(sqrt(3)/2)*2? 
    # No: apothem of unit hex = sqrt(3)/2. Center-to-center for edge-sharing = 2 * apothem = sqrt(3).
    # Wait, that's for sharing a full edge. For disjoint (touching is OK), min center dist = sqrt(3).
    # Actually two unit regular hexagons: the minimum distance between centers for them to be disjoint
    # (non-overlapping) is 2 * apothem = 2*(sqrt(3)/2) = sqrt(3) ≈ 1.732.
    
    # Hex lattice with spacing sqrt(3) (edge-adjacent) or 2 (vertex-adjacent is larger).
    # For a tight packing, use edge-adjacent spacing = sqrt(3).
    
    sqrt3 = np.sqrt(3)
    # Axial to cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    # Pick 11 axial coordinates forming a compact cluster
    axial = [
        (0, 0),    # center
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),  # 6 neighbors
        (2, 0), (-2, 0), (1, 1), (-1, -1),  # 4 more from second ring
    ]
    positions = []
    for q, r in axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        positions.append((x, y))
    
    # Center the cluster
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    positions = [(p[0] - cx, p[1] - cy) for p in positions]
    
    return positions

def _try_rotation_translation(positions, angles_deg, outer_center, outer_angle_deg):
    """Try a specific configuration, return (side, valid)."""
    inner_pos = np.array(positions)
    inner_ang = np.array(angles_deg)
    oc = np.array(outer_center)
    return _evaluate_packing(inner_pos, inner_ang, oc, outer_angle_deg)

def construct():
    global START_TIME
    if _elapsed() > 10:
        START_TIME = time.time()
    
    best_side = 40.0
    best_result = None
    
    # Build base cluster
    base_positions = _build_hex_lattice_positions()
    base_pos = np.array(base_positions)
    base_ang = np.zeros(N)
    
    # Try discrete rotations of the entire cluster (multiples of 30 degrees)
    # and the outer hexagon rotation
    best_found = (40.0, None, None, None, None)
    
    for outer_ang in range(0, 360, 30):
        if _elapsed() > DEADLINE - 10:
            break
        for cluster_rot in range(0, 360, 30):
            if _elapsed() > DEADLINE - 10:
                break
            # Rotate all inner positions and angles
            a = np.deg2rad(cluster_rot)
            cos_a = np.cos(a)
            sin_a = np.sin(a)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            rotated_pos = (R @ base_pos.T).T
            rotated_ang = base_ang + cluster_rot
            rotated_ang = rotated_ang % 360
            
            # Try to center: outer hexagon centered at origin
            oc = (0.0, 0.0)
            
            side, valid = _evaluate_packing(rotated_pos, rotated_ang, oc, outer_ang)
            
            if valid and side < best_found[0]:
                best_found = (side, rotated_pos.copy(), rotated_ang.copy(), oc, outer_ang)
    
    if best_found[1] is not None:
        side = best_found[0]
        pos = best_found[1]
        ang = best_found[2]
        oc = best_found[3]
        oang = best_found[4]
        
        # Build the return format
        inner = [[float(pos[i][0]), float(pos[i][1]), float(ang[i])] for i in range(N)]
        return inner, (float(oc[0]), float(oc[1])), float(side), float(oang)
    
    # Fallback: original seed-like answer
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
