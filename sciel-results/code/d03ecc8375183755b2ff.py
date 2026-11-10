import numpy as np

def hexagon_packing_12():
    # Define unit hexagon vertices (side 1, angle 0)
    # Vertices at angles 0, 60, ... 300
    angles = np.deg2rad(np.arange(0, 360, 60))
    unit_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Define grid vectors for hex tiling
    # Neighbor vectors
    v = [
        np.array([1.5, np.sqrt(3)/2]),
        np.array([0, np.sqrt(3)]),
        np.array([-1.5, np.sqrt(3)/2]),
        np.array([-1.5, -np.sqrt(3)/2]),
        np.array([0, -np.sqrt(3)]),
        np.array([1.5, -np.sqrt(3)/2])
    ]
    
    # Centers of the 7-hexagon cluster
    centers = [np.array([0.0, 0.0])]
    for vec in v:
        centers.append(vec.copy())
        
    # Add 5 more from the second ring
    # Second ring centers: 2*v[i] and v[i]+v[(i+1)%6]
    # Let's pick 5 of the 2*v[i] (the "corners")
    # Indices 0, 1, 2, 3, 4
    for i in range(5):
        centers.append(2 * v[i])
        
    # Now we have 12 centers
    # Generate all vertices of all 12 hexagons
    all_verts = []
    for c in centers:
        # Hexagons are all angle 0
        verts = unit_verts + c
        all_verts.append(verts)
        
    all_verts = np.vstack(all_verts)
    
    # Find optimal outer hexagon angle and side length
    best_s = float('inf')
    best_angle = 0
    
    # Scan angles
    for a_deg in np.arange(0, 60, 1): # 60 deg symmetry
        a_rad = np.deg2rad(a_deg)
        # Normals of the outer hexagon
        # Outer hexagon vertices at a_rad + 60k
        # Normals are at a_rad + 30 + 60k
        normal_angles = a_rad + np.deg2rad(30) + np.deg2rad(60) * np.arange(6)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Calculate required apothem
        # For each vertex, project onto all normals
        # proj = all_verts @ normals.T
        # max_proj = max over all vertices and normals
        # apothem = max_proj
        # s = apothem * 2 / sqrt(3)
        
        projs = all_verts @ normals.T
        max_proj = np.max(projs)
        apothem = max_proj
        s = apothem * 2 / np.sqrt(3)
        
        if s < best_s:
            best_s = s
            best_angle = a_deg
            
    # Construct return data
    inner_hex_data = np.zeros((12, 3))
    for i, c in enumerate(centers):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0.0 # angle of inner hexagons
        
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    return inner_hex_data, outer_hex_data, best_s

