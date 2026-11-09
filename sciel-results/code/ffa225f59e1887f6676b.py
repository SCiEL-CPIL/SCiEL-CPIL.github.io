import numpy as np

N = 12

def construct():
    # Define lattice vectors for unit hexagons (side 1, touching side-to-side)
    # Orientation: Vertex at (1,0)
    # Width 2, Height sqrt(3)
    # Center distance: sqrt(3)
    # Basis:
    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 1.5]) # 3/2 = 1.5

    # Generate a few candidate packings
    candidates = []
    
    # Candidate 1: 7-cluster + 5 extensions
    centers_7 = [
        np.array([0,0]),
        u, v, -u, -v, u-v, -u+v
    ]
    # Ring 2 extensions
    ext_1 = 2*u
    ext_2 = 2*v
    ext_3 = 2*(u-v)
    ext_4 = 2*(-u+v)
    ext_5 = 2*(-u)
    
    centers_c1 = centers_7 + [ext_1, ext_2, ext_3, ext_4, ext_5]
    candidates.append(centers_c1)
    
    # Candidate 2: 4x3 Grid
    # i in 0..3, j in 0..2
    centers_c2 = []
    for i in range(4):
        for j in range(3):
            c = i*u + j*v
            centers_c2.append(c)
    candidates.append(centers_c2)
    
    # Candidate 3: 3x4 Grid (transposed)
    centers_c3 = []
    for i in range(3):
        for j in range(4):
            c = i*u + j*v
            centers_c3.append(c)
    candidates.append(centers_c3)
            
    # Calculate the best outer hexagon side length for each candidate
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = 0.0
    best_angle = 0.0

    # The inner hexagons have side 1.
    # We need to find the min S such that all inner hexagons fit in an outer hexagon of side S.
    # We assume the outer hexagon is centered at (0,0) and aligned with the lattice (angle 0).
    # We can also try rotating the outer hexagon, but for symmetric packings, 0 is good.
    # For asymmetric packings, we might need to rotate.
    # Let's check rotation 0 and 30.
    
    for centers in candidates:
        # Center the packing
        mean = np.mean(centers, axis=0)
        centered = [c - mean for c in centers]
        
        # The vertices of a unit hexagon (side 1, angle 0)
        # Angle 0: vertex at (1,0)
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # For each inner hexagon, get its vertices
        all_points = []
        for c in centered:
            verts = hex_verts + c
            all_points.extend(verts)
        
        all_points = np.array(all_points)
        
        # Check outer hexagon constraints
        # Outer hexagon normals at 0, 60, 120, 180, 240, 300
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Project points onto normals
        # max_proj[k] = max dot product with normal k
        # min_proj[k] = min dot product with normal k
        # Required d >= max(max_proj, -min_proj)
        
        for outer_angle_deg in [0, 30]:
            outer_rot = np.deg2rad(outer_angle_deg)
            # Rotate normals
            c_rot = np.cos(outer_rot)
            s_rot = np.sin(outer_rot)
            rot_mat = np.array([[c_rot, -s_rot], [s_rot, c_rot]])
            rotated_normals = normals @ rot_mat.T
            
            projections = all_points @ rotated_normals.T
            max_projs = np.max(projections, axis=0)
            min_projs = np.min(projections, axis=0)
            
            # Distance from center to side
            d_required = np.max(np.maximum(max_projs, -min_projs))
            
            # Side length S = 2 * d / sqrt(3)
            S = 2 * d_required / np.sqrt(3)
            
            if S < best_score:
                best_score = S
                best_inner = [[float(c[0] + mean[0]), float(c[1] + mean[1]), 0.0] for c in centers]
                best_center = (float(mean[0]), float(mean[1]))
                best_side = float(S)
                best_angle = float(outer_angle_deg)
                
    # If no valid packing found (should not happen), fallback
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END
