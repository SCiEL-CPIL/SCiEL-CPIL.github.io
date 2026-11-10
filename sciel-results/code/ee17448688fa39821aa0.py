# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _min_containing_hex_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum side length s of a regular hexagon (centered at outer_center,
    with the given angle) that contains all inner unit hexagons.
    """
    ov_cx, ov_cy = outer_center[0], outer_center[1]
    ov_a = np.deg2rad(outer_angle_deg)
    
    # The 6 outward normals to the sides of the outer hexagon
    normal_angles = ov_a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # For each inner hexagon, find the maximum projection onto each normal
    max_projections = np.zeros(6)
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        hv_rel = hv - np.array([ov_cx, ov_cy])
        projections = hv_rel @ normals.T
        max_projections = np.maximum(max_projections, np.max(projections, axis=0))
    
    s_min = np.max(max_projections) / (np.sqrt(3.0) / 2.0)
    return s_min


def _check_separation(inner):
    """
    Check if all pairs of inner hexagons are separated using SAT.
    """
    n = len(inner)
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    
    for i in range(n):
        for j in range(i + 1, n):
            v1 = inner_verts[i]
            v2 = inner_verts[j]
            
            edges = []
            for k in range(6):
                edges.append(v1[(k + 1) % 6] - v1[k])
                edges.append(v2[(k + 1) % 6] - v2[k])
            
            separated = False
            for edge in edges:
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                gap1 = min2 - max1
                gap2 = min1 - max2
                if gap1 > 1e-6 or gap2 > 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def _build_rigid_cluster(rotation_deg, center_x, center_y):
    """
    Build a rigid cluster of 12 unit hexagons in a hexagonal close-packed arrangement.
    The cluster is rotated by rotation_deg and translated to (center_x, center_y).
    All hexagons have the same orientation (angle = rotation_deg).
    """
    # Hexagonal close-packed positions for 12 unit hexagons
    # Distance between adjacent centers in same row: 3 (for flat-top)
    # Actually for pointy-top (angle 0 means vertex at right), adjacent centers:
    # Let's use a standard hexagonal lattice.
    
    # For unit hexagons with angle 0 (vertex pointing right):
    # Width = 2, Height = sqrt(3)
    # Adjacent in x: dx = 3, dy = 0 (touching at flat sides? No...)
    
    # Let me reconsider. For a regular hexagon with side 1, angle 0:
    # Vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # This is "pointy-top" if we think of it as vertex at top... actually vertex at angle 0 is at (1,0)
    # So it's "flat-top" in the sense that the top and bottom are flat edges.
    
    # For flat-top hexagons:
    # - Horizontal spacing: 3 (two hexagons side by side touch at vertical edges)
    # - Vertical spacing: sqrt(3) (two hexagons stacked touch at horizontal edges)
    # - Offset rows: dx = 1.5, dy = sqrt(3)
    
    # Let me use a cleaner approach. The distance between centers of two touching
    # unit hexagons depends on orientation. For flat-top (angle 0):
    # - Left-right neighbors: distance = 3 (touching at left/right vertices? No, at edges)
    #   Actually the hexagon has width 2 (from -1 to 1 in x), so two side-by-side 
    #   hexagons with centers at (0,0) and (3,0) touch at x=1 and x=1.
    #   Wait, the rightmost point is at x=1, leftmost at x=-1. So center-to-center = 2?
    #   No: hexagon at (0,0) has right vertex at (1,0). Hexagon at (2,0) has left vertex at (1,0).
    #   They touch at (1,0). But do their edges overlap? The right edge of hex1 goes from 
    #   (0.5, sqrt(3)/2) to (0.5, -sqrt(3)/2)? No, that's not right.
    
    # Let me just compute: vertices of hex at (0,0), side 1, angle 0:
    # v0 = (1, 0)
    # v1 = (0.5, sqrt(3)/2)
    # v2 = (-0.5, sqrt(3)/2)
    # v3 = (-1, 0)
    # v4 = (-0.5, -sqrt(3)/2)
    # v5 = (0.5, -sqrt(3)/2)
    
    # Edge v0-v1: from (1,0) to (0.5, sqrt(3)/2). Midpoint: (0.75, sqrt(3)/4)
    # Edge v5-v0: from (0.5, -sqrt(3)/2) to (1, 0). Midpoint: (0.75, -sqrt(3)/4)
    
    # For two hexagons to touch along the right edge of one and left edge of another:
    # The right "side" of a flat-top hexagon is not a single edge. It has two edges 
    # meeting at vertex v0=(1,0).
    
    # So for flat-top hexagons, the closest approach in the x-direction is at the vertex.
    # Two hexagons centered at (0,0) and (2,0) would have their vertices touching at (1,0).
    # But their edges would overlap! Edge v0-v1 of hex1 and edge v3-v2 of hex2...
    
    # I think for flat-top hexagons, the proper tiling uses:
    # - Same row: dx = 3, dy = 0 (they don't touch, there's a gap)
    # - Actually no. Let me think about honeycomb tiling.
    
    # In a honeycomb tiling of flat-top hexagons:
    # - Horizontal neighbors: dx = 3, dy = 0? No...
    # - The hexagons in a honeycomb tile the plane with no gaps.
    # - For flat-top hexagons, the lattice vectors are:
    #   a1 = (3, 0)  -- no, that would leave gaps
    #   
    # Let me just use known results. For unit hexagons (side length 1) in a honeycomb:
    # The center-to-center distance between adjacent hexagons is 2*sqrt(3)/sqrt(3) = ...
    # 
    # Actually, the inradius of a unit hexagon is sqrt(3)/2. Two hexagons touching 
    # at their edges have center distance = 2 * inradius = sqrt(3).
    # But that's only if they touch at flat edges.
    
    # For flat-top hexagons, the flat edges are top and bottom (horizontal).
    # So two hexagons stacked vertically touch at distance sqrt(3) in y.
    # Two hexagons side by side... they touch at vertices, not edges, if dx=2.
    # 
    # In a proper honeycomb, adjacent hexagons share an edge. For flat-top:
    # - Vertical neighbors: dy = sqrt(3), dx = 0
    # - Diagonal neighbors: dx = 1.5, dy = sqrt(3)/2
    # 
    # Wait, that doesn't seem right either. Let me just use the standard hex grid.
    
    # Standard hex grid with flat-top hexagons:
    # The six neighbors of a hexagon at (0,0) are at:
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), 
    # (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    # 
    # Hmm, I'm getting confused. Let me just use a simple approach:
    # Place 12 hexagons in a roughly circular/hexagonal pattern and verify separation.
    
    # Actually, the simplest rigid cluster: use a hexagonal number pattern.
    # 1 + 6 + 5 = 12 (center + ring 1 + 5 of ring 2)
    # Or 3 + 4 + 3 + 2 = 12 in rows
    
    # Let me use row-based placement for flat-top hexagons:
    # Row y=0: 4 hexagons at x = -4.5, -1.5, 1.5, 4.5
    # Row y=sqrt(3): 4 hexagons at x = -3, 0, 3, 6 -> no, need symmetry
    # 
    # For flat-top hexagons with side 1:
    # - Width = 2 (vertex to vertex horizontally)
    # - Height = sqrt(3) (edge to edge vertically)  
    # - In a row, hexagons are spaced 3 apart (they interlock)
    # - Adjacent rows are offset by 1.5 in x and sqrt(3) in y
    
    # Let me try:
    # Row 0 (y=0): x = -4.5, -1.5, 1.5, 4.5 (4 hexagons)
    # Row 1 (y=sqrt(3)): x = -3, 0, 3 (3 hexagons)  
    # Row 2 (y=2*sqrt(3)): x = -4.5, -1.5, 1.5, 4.5 (4 hexagons)
    # Row 3 (y=3*sqrt(3)): x = 0 (1 hexagon)
    # Total: 4+3+4+1 = 12
    
    # But this isn't very compact. Let me try a more symmetric arrangement.
    
    # Better: hexagonal cluster
    # Center: (0, 0)
    # Ring 1 (6 hexagons): at distance sqrt(3) in 6 directions (30, 90, 150, 210, 270, 330 degrees)
    #   for flat-top, the "flat" directions are at 90 and 270 (top/bottom)
    #   and the "pointy" directions are at 0, 60, 120, 180, 240, 300
    #   Adjacent hexagons share edges, so center distance = 2 * inradius = sqrt(3)
    #   The 6 edge normals are at 30, 90, 150, 210, 270, 330 degrees
    #   So ring 1 centers are at:
    #   (sqrt(3)*cos(30), sqrt(3)*sin(30)) = (1.5, sqrt(3)/2)
    #   (sqrt(3)*cos(90), sqrt(3)*sin(90)) = (0, sqrt(3))
    #   (sqrt(3)*cos(150), sqrt(3)*sin(150)) = (-1.5, sqrt(3)/2)
    #   (sqrt(3)*cos(210), sqrt(3)*sin(210)) = (-1.5, -sqrt(3)/2)
    #   (sqrt(3)*cos(270), sqrt(3)*sin(270)) = (0, -sqrt(3))
    #   (sqrt(3)*cos(330), sqrt(3)*sin(330)) = (1.5, -sqrt(3)/2)
    # 
    # Ring 2 (5 hexagons): at distance 2*sqrt(3) in some directions
    #   (2*sqrt(3)*cos(90), 2*sqrt(3)*sin(90)) = (0, 2*sqrt(3))
    #   (2*sqrt(3)*cos(150), 2*sqrt(3)*sin(150)) = (-3, sqrt(3))
    #   (2*sqrt(3)*cos(210), 2*sqrt(3)*sin(210)) = (-3, -sqrt(3))
    #   (2*sqrt(3)*cos(270), 2*sqrt(3)*sin(270)) = (0, -2*sqrt(3))
    #   (2*sqrt(3)*cos(330), 2*sqrt(3)*sin(330)) = (3, -sqrt(3))
    #   That's 5, but we need to check if they're all valid.
    
    # Actually let me just use a simple grid and let the optimizer handle it.
    # The key insight is: use a FIXED topology (rigid body) and only optimize 
    # rotation and translation.
    
    # Simple rigid cluster: 12 hexagons in a 3x4 grid pattern (approximately)
    # For flat-top hexagons:
    # Spacing: dx = 3, dy = sqrt(3), with offset rows
    
    sqrt3 = np.sqrt(3.0)
    
    # Positions in local frame (before rotation/translation):
    # Using a hexagonal close-packed arrangement
    local_positions = np.array([
        [0.0, 0.0],           # 0: center
        [3.0, 0.0],           # 1: right
        [-3.0, 0.0],          # 2: left
        [1.5, sqrt3],         # 3: top-right
        [-1.5, sqrt3],        # 4: top-left
        [1.5, -sqrt3],        # 5: bottom-right
        [-1.5, -sqrt3],       # 6: bottom-left
        [4.5, sqrt3],         # 7: far top-right
        [-4.5, sqrt3],        # 8: far top-left
        [4.5, -sqrt3],        # 9: far bottom-right
        [-4.5, -sqrt3],       # 10: far bottom-left
        [0.0, 2*sqrt3],       # 11: far top
    ])
    
    # Rotate positions
    theta = np.deg2rad(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    rotated_x = local_positions[:, 0] * cos_t - local_positions[:, 1] * sin_t
    rotated_y = local_positions[:, 0] * sin_t + local_positions[:, 1] * cos_t
    
    # Translate
    positions_x = rotated_x + center_x
    positions_y = rotated_y + center_y
    
    inner = np.zeros((12, 3))
    inner[:, 0] = positions_x
    inner[:, 1] = positions_y
    inner[:, 2] = rotation_deg  # All hexagons have the same orientation
    
    return inner


def _check_separation_rigid(inner):
    """Check separation for the rigid cluster."""
    return _check_separation(inner)


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon.
    Returns (inner, outer, s).
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    # First, verify our rigid cluster is valid (no overlaps)
    test_inner = _build_rigid_cluster(0.0, 0.0, 0.0)
    if not _check_separation_rigid(test_inner):
        # If the rigid cluster has overlaps, fall back to free optimization
        return _free_optimization_fallback(start_time, deadline)
    
    # Optimize only 3 parameters: rotation angle, center_x, center_y
    # Objective: minimize the containing hexagon side length
    
    def objective(params):
        rotation_deg = params[0]
        center_x = params[1]
        center_y = params[2]
        
        inner = _build_rigid_cluster(rotation_deg, center_x, center_y)
        
        outer_center = np.array([0.0, 0.0])
        outer_angle = 0.0
        
        s = _min_containing_hex_side(inner, outer_center, outer_angle)
        return s
    
    # Try multiple initial rotation angles
    best_params = None
    best_s = np.inf
    
    for init_rot in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0]:
        if time.time() - start_time > deadline - 30.0:
            break
        
        init_params = np.array([init_rot, 0.0, 0.0])
        
        try:
            result = minimize(
                objective,
                init_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Also try Powell for refinement
    if best_params is not None and time.time() - start_time < deadline - 60.0:
        try:
            result = minimize(
                objective,
                best_params,
                method='Powell',
                options={'maxiter': 2000, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback
        return _free_optimization_fallback(start_time, deadline)
    
    # Construct final result
    inner = _build_rigid_cluster(best_params[0], best_params[1], best_params[2])
    
    # Verify separation
    if not _check_separation(inner):
        return _free_optimization_fallback(start_time, deadline)
    
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    s_final = _min_containing_hex_side(inner, outer_center, outer_angle) + 1e-4
    
    outer = np.array([0.0, 0.0, 0.0])
    
    return inner, outer, s_final


def _free_optimization_fallback(start_time, deadline):
    """Fallback: free optimization of all 24 positions."""
    
    def _objective(params, outer_center, outer_angle_deg):
        inner = np.zeros((12, 3))
        inner[:, 0] = params[0::2]
        inner[:, 1] = params[1::2]
        inner[:, 2] = 0.0
        
        if not _check_separation(inner):
            return 1000.0
        
        s = _min_containing_hex_side(inner, outer_center, outer_angle_deg)
        return s
    
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    d = np.sqrt(3.0)
    initial_params = np.array([
        0.0, 0.0,
        d, 0.0,
        -d, 0.0,
        d/2, d*1.5,
        -d/2, d*1.5,
        d/2, -d*1.5,
        -d/2, -d*1.5,
        2*d, 0.0,
        -2*d, 0.0,
        3*d/2, d*1.5,
        -3*d/2, d*1.5,
        0.0, -2*d*1.5,
    ])
    
    inner_init = np.zeros((12, 3))
    inner_init[:, 0] = initial_params[0::2]
    inner_init[:, 1] = initial_params[1::2]
    
    if not _check_separation(inner_init):
        spacing = 2.0 * np.sqrt(3.0) + 1e-3
        initial_params = np.array([])
        for i in range(12):
            initial_params = np.append(initial_params, [i * spacing, 0.0])
        inner_init = np.zeros((12, 3))
        inner_init[:, 0] = initial_params[0::2]
        inner_init[:, 1] = initial_params[1::2]
    
    best_params = initial_params.copy()
    best_s = _min_containing_hex_side(inner_init, outer_center, outer_angle)
    
    try:
        result = minimize(
            _objective,
            initial_params,
            args=(outer_center, outer_angle),
            method='Nelder-Mead',
            options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < best_s:
            best_params = result.x
            best_s = result.fun
    except Exception:
        pass
    
    if time.time() - start_time < deadline - 120.0:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(outer_center, outer_angle),
                method='Powell',
                options={'maxiter': 3000, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            if result.fun < best_s:
                best_params = result.x
                best_s = result.fun
        except Exception:
            pass
    
    inner = np.zeros((12, 3))
    inner[:, 0] = best_params[0::2]
    inner[:, 1] = best_params[1::2]
    inner[:, 2] = 0.0
    
    s_final = _min_containing_hex_side(inner, outer_center, outer_angle) + 1e-4
    
    if not _check_separation(inner):
        inner = inner_init.copy()
        s_final = _min_containing_hex_side(inner, outer_center, outer_angle) + 1e-4
    
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, s_final


# EVOLVE-BLOCK-END
