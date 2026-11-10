# EVOLVE-BLOCK-START
import numpy as np
import time

def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon using a hexagonal lattice
    representation with rotation optimization.
    
    Returns:
        inner: (12, 3) array of (x, y, angle_deg) for unit hexagons
        outer: (3,) array of (x, y, angle_deg) for the outer hexagon
        s: side length of the outer hexagon
    """
    start_time = time.time()
    DEADLINE = 620.0
    
    # Define 12 unit hexagons on a hexagonal lattice (axial coordinates)
    # Pointy-top hexagons, side=1
    # Axial to cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    axial_coords = []
    for r in range(3):
        for q in range(4):
            axial_coords.append((q, r))
    
    axial_coords = np.array(axial_coords, dtype=float)
    
    # Convert to cartesian
    q = axial_coords[:, 0]
    r = axial_coords[:, 1]
    base_centers = np.column_stack([np.sqrt(3) * (q + r / 2.0), 1.5 * r])
    
    # Center at origin
    base_centers -= np.mean(base_centers, axis=0)
    
    def compute_min_containing_hex(centers, inner_angle_rad):
        """
        Find the minimal regular hexagon containing all vertices of the 12 unit hexagons.
        Returns (outer_center, outer_angle_rad, side_length)
        """
        # Get all 72 vertices
        all_verts = []
        for i in range(12):
            cx, cy = centers[i]
            for j in range(6):
                theta = inner_angle_rad + j * np.pi / 3
                all_verts.append([cx + np.cos(theta), cy + np.sin(theta)])
        all_verts = np.array(all_verts)
        
        best_s = float('inf')
        best_oc = None
        best_oa = None
        
        # Try different outer hexagon orientations
        for oa in np.arange(0, np.pi, np.pi / 18):
            # Normals for the 6 edges
            normals = np.array([[np.cos(oa + np.pi/6 + k*np.pi/3), 
                                  np.sin(oa + np.pi/6 + k*np.pi/3)] for k in range(6)])
            
            # The constraint for a point p to be inside hexagon with center c, side s:
            # dot(p - c, normal_k) <= s * sqrt(3)/2 for all k
            # So s >= (2/sqrt(3)) * max_k dot(p - c, normal_k)
            # To minimize s, choose c to minimize max over all p of max_k dot(p-c, normal_k)
            
            # This is: minimize_c max_p max_k [dot(p, n_k) - dot(c, n_k)]
            # = minimize_c max_p max_k [dot(p, n_k)] - dot(c, n_k)
            
            # Let f_p(k) = dot(p, n_k). We want to minimize max_p max_k f_p(k) - dot(c, n_k)
            # For fixed c, the required s is (2/sqrt(3)) * max_p [max_k (dot(p, n_k) - dot(c, n_k))]
            # = (2/sqrt(3)) * [max_p max_k dot(p, n_k) - min over... ]
            
            # Actually: max_p max_k [dot(p, n_k) - dot(c, n_k)]
            # = max_p [max_k dot(p, n_k) - dot(c, n_k)]  (since dot(c,n_k) doesn't depend on p)
            # = max_p [g(p) - dot(c, n_k)]  but n_k varies...
            
            # Let me just compute it directly for a candidate center
            # Optimal center: the one that equalizes the max projections
            # Simple approach: use the center of the bounding box in each normal direction
            
            # For each normal, find the max and min projection
            projections = all_verts @ normals.T  # (72, 6)
            
            # The hexagon must contain all points, so for each normal k:
            # max_p dot(p, n_k) - dot(c, n_k) <= s*sqrt(3)/2
            # dot(c, n_k) - min_p dot(p, n_k) <= s*sqrt(3)/2
            # So s*sqrt(3)/2 >= max_k [max_p dot(p,n_k) - dot(c,n_k)]
            # and s*sqrt(3)/2 >= max_k [dot(c,n_k) - min_p dot(p,n_k)]
            
            # To minimize s, we want to choose c to balance these.
            # The optimal c satisfies: for the critical normals, the margins are equal.
            
            # Simple heuristic: c at the center of the "bounding hexagon"
            # For each normal, the midpoint between max and min projection
            midpoints = []
            for k in range(6):
                mid = (np.max(projections[:, k]) + np.min(projections[:, k])) / 2
                midpoints.append(mid)
            midpoints = np.array(midpoints)
            
            # Solve for c such that dot(c, n_k) = midpoint_k for all k
            # This is a 6x2 system (overdetermined), use least squares
            c, _, _, _ = np.linalg.lstsq(normals.T, midpoints, rcond=None)
            
            # Compute required s
            rel = all_verts - c
            dots = rel @ normals.T
            max_dots = np.max(dots, axis=1)
            required_s = (2.0 / np.sqrt(3.0)) * np.max(max_dots)
            
            if required_s < best_s:
                best_s = required_s
                best_oc = c
                best_oa = oa
        
        return best_oc, best_oa, best_s
    
    # Optimize the rotation angle of the inner configuration
    best_angle = 0
    best_result = None
    best_s = float('inf')
    
    # Coarse search
    for angle in np.linspace(0, np.pi, 37):
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        rotated = base_centers @ R.T
        oc, oa, s = compute_min_containing_hex(rotated, angle)
        if s < best_s:
            best_s = s
            best_angle = angle
            best_result = (rotated, oc, oa)
        
        if time.time() - start_time > DEADLINE:
            break
    
    # Fine search
    if time.time() - start_time < DEADLINE:
        for angle in np.linspace(best_angle - 0.05, best_angle + 0.05, 31):
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
            rotated = base_centers @ R.T
            oc, oa, s = compute_min_containing_hex(rotated, angle)
            if s < best_s:
                best_s = s
                best_angle = angle
                best_result = (rotated, oc, oa)
    
    # Even finer
    if time.time() - start_time < DEADLINE:
        for angle in np.linspace(best_angle - 0.01, best_angle + 0.01, 21):
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
            rotated = base_centers @ R.T
            oc, oa, s = compute_min_containing_hex(rotated, angle)
            if s < best_s:
                best_s = s
                best_angle = angle
                best_result = (rotated, oc, oa)
    
    final_centers, outer_center, outer_angle = best_result
    s = best_s
    
    # Construct output
    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = final_centers[i, 0]
        inner[i, 1] = final_centers[i, 1]
        inner[i, 2] = np.degrees(best_angle)
    
    outer = np.array([outer_center[0], outer_center[1], np.degrees(outer_angle)])
    
    return inner, outer, s

# EVOLVE-BLOCK-END
