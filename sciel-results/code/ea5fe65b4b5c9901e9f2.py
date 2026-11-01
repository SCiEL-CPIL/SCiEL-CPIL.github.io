import numpy as np
import science_chemistry

N = 55

# EVOLVE-BLOCK-START
def construct():
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize vertices to unit length
    norms = np.linalg.norm(verts, axis=1)
    verts_unit = verts / norms[:, np.newaxis]
    
    # Find edges: pairs of vertices with minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            dists[i, j] = d
            dists[j, i] = d
            
    # The edge length is the minimum non-zero distance
    min_dist = np.min(dists[dists > 1e-6])
    
    # Collect edge midpoints
    midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-6:
                    midpoints.append(mid / mid_norm)
                    
    midpoints = np.array(midpoints, dtype=float)
    
    # Initial radii guesses
    R_in = 1.1
    R_out = 2.0
    
    # Construct positions
    pos = []
    # Center
    pos.append([0.0, 0.0, 0.0])
    
    # Inner shell (12 atoms)
    for v in verts_unit:
        pos.append(R_in * v)
        
    # Outer shell vertices (12 atoms)
    for v in verts_unit:
        pos.append(R_out * v)
        
    # Outer shell edge midpoints (30 atoms)
    for m in midpoints:
        pos.append(R_out * m)
        
    positions = np.array(pos, dtype=float)
    
    # Relax using L-BFGS
    relaxed_positions = science_chemistry.relax(positions)
    
    return relaxed_positions
# EVOLVE-BLOCK-END
