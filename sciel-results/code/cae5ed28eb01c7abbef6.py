import numpy as np

def hexagon_packing_11():
    # Configuration: 7 core + 4 outer
    # Rotated by 30 degrees to align outer points with core gaps?
    # Actually, at 30 deg rotation:
    # Core points at 30, 90, ...
    # Outer points at 0, 60, 180, 240
    
    # Scale factor to ensure separation
    scale = 1.0 + 1e-6
    
    r_core = np.sqrt(3) * scale
    r_outer = 3.0 * scale
    
    # Core angles
    angles_core = np.deg2rad([30, 90, 150, 210, 270, 330])
    centers_core = np.array([
        [r_core * np.cos(a), r_core * np.sin(a)] for a in angles_core
    ])
    
    # Outer angles
    angles_outer = np.deg2rad([0, 60, 180, 240])
    centers_outer = np.array([
        [r_outer * np.cos(a), r_outer * np.sin(a)] for a in angles_outer
    ])
    
    centers = np.vstack([centers_core, centers_outer])
    
    # All inner hexagons have angle 0
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = centers[:, 0]
    inner_hex_data[:, 1] = centers[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    # Calculate optimal outer side length
    # S = (2/sqrt(3)) * max_proj + 1
    # max_proj is max over all vertices of all hexagons projected on outer normals
    
    # Outer normals
    normal_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.array([
        [np.cos(a), np.sin(a)] for a in normal_angles
    ])
    
    # Inner hexagon vertices (angle 0)
    # Vertices at 0, 60, 120, 180, 240, 300
    hex_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_verts = np.array([
        [np.cos(a), np.sin(a)] for a in hex_angles
    ])
    
    max_proj = 0
    for c in centers:
        verts = c + hex_verts
        # Project onto normals
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
        
    side_length = (2.0 / np.sqrt(3.0)) * max_proj + 1.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, side_length
