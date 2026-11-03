import numpy as np

N = 11

def get_hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    # Pointy top hexagon vertices
    angles = np.array([90, 150, 210, 270, 330, 30]) + angle_deg
    angles_rad = np.deg2rad(angles)
    x = cx + np.cos(angles_rad)
    y = cy + np.sin(angles_rad)
    return np.column_stack((x, y))

def min_enclosing_hex_side(vertices):
    # vertices: (M, 2) array
    # Returns (minimal_side, best_alpha_deg)
    best_s = float('inf')
    best_alpha = 0.0
    # Search alpha in [0, 30) with fine steps
    # 30 degrees is 1/2 of symmetry period.
    # Let's use 60 steps for 30 deg => 0.5 deg resolution
    for alpha_deg in np.linspace(0, 30, 60, endpoint=False):
        alpha_rad = np.deg2rad(alpha_deg)
        # Normals to sides of the outer hexagon
        # Sides are perpendicular to normals.
        # If outer hexagon is rotated by alpha, its vertices are at alpha + k*60.
        # Its sides are at alpha + 30 + k*60.
        # Normals point outward at these angles.
        normal_angles = np.deg2rad(alpha_deg + 30 + np.arange(6)*60)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        proj = vertices @ normals.T
        max_proj = np.max(np.abs(proj))
        s = max_proj * 2.0 / np.sqrt(3.0)
        
        if s < best_s:
            best_s = s
            best_alpha = alpha_deg
            
    return best_s, best_alpha

def construct():
    # Lattice basis
    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    def lattice_to_xy(i, j):
        return i*u + j*v

    # Candidate index sets
    idx_sets = [
        # 4-3-2-1-1
        [(0,0), (1,0), (2,0), (3,0), 
         (0,1), (1,1), (2,1),
         (0,2), (1,2),
         (0,3), (3,1)],
        # 4-4-3
        [(0,0), (1,0), (2,0), (3,0),
         (0,1), (1,1), (2,1), (3,1),
         (0,2), (1,2), (2,2)],
        # 3-4-3-1
        [(0,0), (1,0), (2,0),
         (0,1), (1,1), (2,1), (3,1),
         (0,2), (1,2), (2,2),
         (0,3), (3,0)],
        # 2-3-4-2
        [(0,0), (1,0),
         (0,1), (1,1), (2,1),
         (0,2), (1,2), (2,2), (3,2),
         (0,3), (1,3)]
    ]
    
    best_s = float('inf')
    best_inner = None
    best_outer_angle = 0.0
    
    for idx in idx_sets:
        centers = np.array([lattice_to_xy(i, j) for i, j in idx])
        centroid = np.mean(centers, axis=0)
        centers -= centroid
        
        # Try inner rotation 0 and 30
        for inner_rot in [0.0, 30.0]:
            all_vertices = []
            for c in centers:
                verts = get_hex_vertices(c[0], c[1], inner_rot)
                all_vertices.append(verts)
            all_vertices = np.vstack(all_vertices)
            
            s, alpha = min_enclosing_hex_side(all_vertices)
            
            if s < best_s:
                best_s = s
                best_inner = [[c[0], c[1], inner_rot] for c in centers]
                best_outer_angle = alpha
                
    if best_inner is None:
        # Fallback to baseline
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, (0.0, 0.0), best_s, best_outer_angle
