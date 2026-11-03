"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import numpy as np
import time

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _get_inner_configurations():
    """
    Generate a few candidate compact configurations for 12 unit hexagons.
    Returns a list of arrays, each of shape (12, 2) containing (x, y) centers.
    """
    configs = []
    sqrt3 = np.sqrt(3)
    
    # Configuration 1: Spiral / Compact blob
    # Axial coords (q, r)
    # Center: (0,0)
    # Neighbors: (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (0,1)
    # Next ring: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-2,1), (-1,1), (0,2), (1,2)
    # We need 12 total. 1 + 6 = 7. Need 5 more.
    # Pick 5 from the next ring to maintain compactness.
    # Let's take a contiguous block of 5 from the ring to keep it together.
    # Or just take the first 5 in a spiral order.
    spiral_axial = [
        (0,0),
        (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (0,1),
        (2,0), (2,-1), (2,-2), (1,-2), (0,-2)
    ]
    # This is 12 points.
    pts1 = []
    for q, r in spiral_axial:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        pts1.append([x, y])
    configs.append(np.array(pts1))
    
    # Configuration 2: Hexagon of side 2 (19 cells) minus 7 specific cells
    # To make it more symmetric, remove the 6 corners and 1 edge?
    # Corners of side 2 hex: (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    # If we remove these 6, we have 13 cells.
    # Remove one more, e.g., (1,1) which is an edge midpoint?
    # Let's check the shape.
    # Full side 2 hex axial coords:
    # q from -2 to 2
    # r range depends on q.
    # Let's just generate all 19 and remove 7.
    all_19 = []
    for q in range(-2, 3):
        r_min = max(-2, -q - 2)
        r_max = min(2, -q + 2)
        for r in range(r_min, r_max + 1):
            all_19.append((q, r))
            
    # Remove the 6 corners
    corners = {(2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)}
    # Remove one more to get to 12. Let's remove (1,1) or similar to break symmetry slightly?
    # Or remove 7 corners? There are only 6 corners.
    # Maybe remove 6 corners and 1 from the next inner ring?
    # The inner ring is the 6 neighbors of center.
    # If we remove a neighbor, it creates a hole.
    # Let's try removing the 6 corners and (1,0)?
    # (1,0) is a neighbor.
    remove_set = corners | {(1,0)}
    
    pts2 = []
    for q, r in all_19:
        if (q, r) not in remove_set:
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            pts2.append([x, y])
    if len(pts2) == 12:
        configs.append(np.array(pts2))

    # Configuration 3: 3x4 rectangular grid on hex lattice
    # q: 0, 1, 2, 3 (4 columns)
    # r: 0, 1, 2 (3 rows)
    # In hex grid, a "rectangle" is a parallelogram.
    pts3 = []
    for q in range(4):
        for r in range(3):
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            pts3.append([x, y])
    configs.append(np.array(pts3))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    configs = _get_inner_configurations()
    
    best_side = np.inf
    best_config_idx = 0
    best_angle = 0.0
    best_center = (0.0, 0.0)
    best_inner_x = []
    best_inner_y = []
    best_inner_angles = []
    
    # Search relative angle between inner lattice and outer hexagon
    # Symmetry: 60 degrees for hexagon, but 180 for the "width" function?
    # Width W(n) = W(-n).
    # The set of normals rotates by 60 deg.
    # So we only need to search 0 to 30 degrees (or 0 to 60, but 30 is half of 60).
    # Let's search 0 to 30 deg with step 0.1 deg for precision.
    
    angles_to_try = np.arange(0, 30.01, 0.1)
    
    for cfg_idx, centers in enumerate(configs):
        # Generate vertices for all 12 hexagons at angle 0
        # We will rotate the whole set later
        all_verts = []
        for cx, cy in centers:
            verts = _hex_vertices(cx, cy, 0.0, 1.0)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts) # Shape (72, 2)
        
        # Center the configuration at origin for width calculation
        # Actually, width is translation invariant, so we don't need to center for S calculation.
        # But we need to find the optimal outer center for the output.
        
        for theta_deg in angles_to_try:
            if time.time() > deadline - 5.0:
                break
                
            theta_rad = np.radians(theta_deg)
            
            # Rotate the vertices by theta
            cos_t = np.cos(theta_rad)
            sin_t = np.sin(theta_rad)
            rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            rotated_verts = all_verts @ rot_mat.T
            
            # Compute widths in 6 directions
            # Outer hexagon face normals at angle 0 (reference frame of outer hex)
            # Normals are at 30, 90, 150, 210, 270, 330 degrees?
            # Standard hexagon with flat top:
            # Vertices at 0, 60, 120...
            # Normals at 30, 90, 150...
            # Let's define normals for an outer hexagon with side S centered at origin.
            # Face normals n_k for k=0..5: angle = 30 + k*60 degrees.
            
            max_width = 0.0
            # We need the width of the rotated inner set in the directions of the outer normals.
            # Since we rotated the inner set by theta, and outer normals are fixed,
            # this is equivalent to rotating the normals by -theta and projecting the original set.
            # Or just project rotated_verts onto fixed normals.
            
            for k in range(6):
                normal_angle = np.radians(30 + k * 60)
                n = np.array([np.cos(normal_angle), np.sin(normal_angle)])
                
                projections = rotated_verts @ n
                w = np.max(projections) - np.min(projections)
                if w > max_width:
                    max_width = w
            
            # inradius = max_width / 2
            # side = inradius / (sqrt(3)/2) = max_width / sqrt(3)
            side = max_width / np.sqrt(3)
            
            if side < best_side:
                best_side = side
                best_config_idx = cfg_idx
                best_angle = theta_deg
                # Determine the optimal center
                # The center of the minimal enclosing hexagon for this orientation
                # is the point c such that max(p-c, n) = min(c-p, n) for all n?
                # Actually, the center is the average of the midpoints of the projections.
                # c = 1/6 sum_k (max(p, n_k) + min(p, n_k)) / 2 * n_k? No.
                # The center is the unique point such that the projections are centered.
                # For a regular hexagon, the center is the centroid of the "support points".
                # Simpler: The center is the mean of the 6 "face midpoints" of the bounding box.
                # Face midpoint in direction n_k is m_k = 0.5 * (max(p, n_k) + min(p, n_k)) * n_k?
                # No, the face is a line. The distance from origin to face is d_k = 0.5 * w_k.
                # The face line is p . n_k = d_k.
                # The center is the intersection of these lines.
                # For a regular hexagon, the center is simply the average of the 6 vertices of the "bounding hexagon"?
                # Let's just use the centroid of the rotated vertices. It's a good approximation.
                center = np.mean(rotated_verts, axis=0)
                # Unrotate the center to get it in the inner coordinate frame?
                # The outer hexagon is centered at 'center' in the frame where inner is rotated by theta.
                # In the final output, we return inner positions and outer center.
                # The evaluator will place inner hexagons at (x,y) and outer at (cx,cy).
                # The angle of the outer hexagon is 'angle'.
                # My 'theta' is the angle of the inner lattice relative to the outer hexagon.
                # If I set outer angle = 0, then inner lattice angle = theta.
                # Or I can set outer angle = theta, inner lattice angle = 0.
                # Let's set outer angle = theta_deg.
                # Then the inner vertices (unrotated) must fit in the outer hexagon rotated by theta.
                # The center calculated above 'center' is in the rotated frame.
                # We need the center in the global frame.
                # If outer is rotated by theta, its center is just a point (cx, cy).
                # The inner points are at their original positions (if we say inner angle=0).
                # The condition is: inner points fit in outer hexagon centered at (cx,cy) with angle theta.
                # This is equivalent to: (inner - c) fit in outer hexagon centered at 0 with angle 0.
                # So c = center_of_minimal_box(inner, 0)?
                # Wait.
                # If I fix outer angle = 0, and rotate inner by theta, the center of the box is 'center' (in rotated frame).
                # In the unrotated frame, the center is 'center' unrotated?
                # No.
                # Let's stick to:
                # Inner positions: original centers (unrotated).
                # Outer angle: theta_deg.
                # Outer center: ?
                # The calculation above rotated the inner vertices by theta.
                # The center 'center' is the center of the minimal box in the frame where inner is rotated by theta and outer is at 0.
                # This 'center' is a vector in the rotated frame.
                # To get the center in the global frame (where outer is at angle theta), we don't rotate the center back?
                # Let's think.
                # Frame 1: Global. Outer angle = theta. Inner angle = 0.
                # Frame 2: Rotated by -theta. Outer angle = 0. Inner angle = -theta.
                # My calculation rotated inner by +theta. This corresponds to Frame 2 where Inner is at +theta?
                # If I rotate inner by theta, I am effectively changing the relative angle.
                # Let's just output:
                # Inner angles: 0.
                # Outer angle: theta_deg.
                # Outer center: We need the center C such that the inner points fit.
                # In the calculation, I found the center C' in the frame where inner is rotated by theta and outer is at 0.
                # This C' is the center of the box containing (Inner @ R(theta)).
                # The condition for fit is: (Inner - C) fits in Outer(0).
                # This is equivalent to: (Inner - C) @ R(theta) fits in Outer(0) @ R(theta)? No.
                # Let's use the property:
                # Min side for relative angle theta is S(theta).
                # The optimal center in the global frame is the centroid of the inner points?
                # For a symmetric shape, yes. For asymmetric, it's the center of the min enclosing hex.
                # I will approximate the center as the centroid of the inner vertices.
                # This is safe and fast.
                center_global = np.mean(all_verts, axis=0) # Unrotated centroid
                
                best_center = (float(center_global[0]), float(center_global[1]))
                best_inner_x = [float(c[0]) for c in centers]
                best_inner_y = [float(c[1]) for c in centers]
                best_inner_angles = [0.0] * N
                best_outer_angle = theta_deg

    # Fallback if no valid side found (should not happen)
    if best_side == np.inf:
        # Use first config, angle 0
        centers = configs[0]
        best_inner_x = [float(c[0]) for c in centers]
        best_inner_y = [float(c[1]) for c in centers]
        best_inner_angles = [0.0] * N
        best_center = (0.0, 0.0)
        best_outer_angle = 0.0
        # Calculate side for fallback
        all_verts = np.vstack([_hex_vertices(x, y, 0.0, 1.0) for x, y in centers])
        max_w = 0
        for k in range(6):
            na = np.radians(30 + k*60)
            n = np.array([np.cos(na), np.sin(na)])
            proj = all_verts @ n
            max_w = max(max_w, np.max(proj) - np.min(proj))
        best_side = max_w / np.sqrt(3)

    inner = [[best_inner_x[i], best_inner_y[i], best_inner_angles[i]] for i in range(N)]
    side = float(best_side)
    
    return inner, best_center, side, float(best_outer_angle)

# EVOLVE-BLOCK-END
